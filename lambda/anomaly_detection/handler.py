import json
import boto3
import os
import time
from datetime import datetime
from decimal import Decimal
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
athena_client = boto3.client('athena')
s3_client = boto3.client('s3')
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Environment variables
ATHENA_DATABASE = os.environ.get('ATHENA_DATABASE', 'cms_open_payments')
ATHENA_TABLE = os.environ.get('ATHENA_TABLE', 'general_payments')
FEATURE_STORE_DATABASE = os.environ.get('FEATURE_STORE_DATABASE', 'sagemaker_featurestore')
FEATURE_STORE_TABLE = os.environ.get('FEATURE_STORE_TABLE', '')
ATHENA_OUTPUT_BUCKET = os.environ.get('ATHENA_OUTPUT_BUCKET')
SAGEMAKER_ENDPOINT = os.environ.get('SAGEMAKER_ENDPOINT')

# Model feature names in exact order expected by the model (16 features)
MODEL_FEATURES = [
    'total_amount_of_payment_usdollars',
    'number_of_payments_included_in_total_amount',
    'payment_year',
    'payment_month',
    'payment_quarter',
    'payment_dayofweek',
    'is_weekend',
    'hist_pay_count',
    'hist_pay_total',
    'hist_pay_avg',
    'hist_pay_std',
    'hist_pay_max',
    'amt_to_avg_ratio',
    'amt_to_max_ratio',
    'is_new_recipient',
    'is_high_risk_nature'
]

# Features that actually exist in the feature store (8 features + 2 metadata)
FEATURE_STORE_FEATURES = [
    'total_amount_of_payment_usdollars',
    'payment_year',
    'hist_pay_avg',
    'hist_pay_count',
    'hist_pay_std',
    'hist_pay_max',
    'amt_to_max_ratio',
    'is_high_risk_nature',
    'nature_of_payment_or_transfer_of_value',
    'recipient_state'
]


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder for Decimal types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


def query_athena(query, database):
    """Execute Athena query and return results"""
    try:
        # Start query execution
        response = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': database},
            ResultConfiguration={
                'OutputLocation': f's3://{ATHENA_OUTPUT_BUCKET}/query-results/'
            }
        )
        
        query_execution_id = response['QueryExecutionId']
        logger.info(f"Query execution started: {query_execution_id}")
        
        # Wait for query to complete
        max_attempts = 60
        for attempt in range(max_attempts):
            query_status = athena_client.get_query_execution(
                QueryExecutionId=query_execution_id
            )
            
            status = query_status['QueryExecution']['Status']['State']
            
            if status == 'SUCCEEDED':
                break
            elif status in ['FAILED', 'CANCELLED']:
                reason = query_status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
                raise Exception(f"Query {status}: {reason}")
            
            time.sleep(2)
        
        if status != 'SUCCEEDED':
            raise Exception("Query execution timeout")
        
        # Get query results
        result_paginator = athena_client.get_paginator('get_query_results')
        result_iterator = result_paginator.paginate(
            QueryExecutionId=query_execution_id
        )
        
        results = []
        for result_page in result_iterator:
            for row in result_page['ResultSet']['Rows'][1:]:  # Skip header
                results.append([col.get('VarCharValue', '') for col in row['Data']])
        
        # Get column names
        columns = [col['Label'] for col in 
                   result_page['ResultSet']['ResultSetMetadata']['ColumnInfo']]
        
        return columns, results
        
    except Exception as e:
        logger.error(f"Athena query error: {str(e)}")
        raise


def prepare_features(records, columns):
    """Prepare features for SageMaker model"""
    features_list = []
    
    for record in records:
        record_dict = dict(zip(columns, record))
        
        try:
            # Extract features in the exact order expected by the model
            # For features not in feature store, use default values (0.0)
            # Return as dictionary with feature names as keys
            feature_dict = {}
            for feature_name in MODEL_FEATURES:
                value = record_dict.get(feature_name, 0)
                # Convert to float, handling None and empty values
                try:
                    feature_dict[feature_name] = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    feature_dict[feature_name] = 0.0
            
            features_list.append(feature_dict)
        except Exception as e:
            logger.warning(f"Error preparing features for record: {str(e)}")
            continue
    
    return features_list


def call_sagemaker_endpoint(features):
    """Call SageMaker endpoint for inference"""
    try:
        # Prepare payload - features is a list of dicts with feature names as keys
        payload = json.dumps(features)
        
        logger.info(f"Invoking SageMaker endpoint with {len(features)} samples")
        logger.debug(f"Sample payload (first record): {list(features[0].keys()) if features else 'empty'}")
        
        # Invoke endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        logger.info(f"SageMaker response received successfully")
        return result
        
    except Exception as e:
        logger.error(f"SageMaker endpoint error: {str(e)}")
        raise


def process_inference_results(records, columns, predictions):
    """Process and combine records with predictions"""
    results = []
    
    for i, record in enumerate(records):
        if i >= len(predictions):
            break
            
        record_dict = dict(zip(columns, record))
        prediction = predictions[i]
        
        # Extract the prediction result
        result = {
            'record': {
                'Profile_ID': record_dict.get('covered_recipient_profile_id', ''),
                'Total_Amount_of_Payment_USDollars': float(record_dict.get('total_amount_of_payment_usdollars', 0)),
                'Payment_Year': int(float(record_dict.get('payment_year', 0))),
                'Nature_of_Payment': record_dict.get('nature_of_payment_or_transfer_of_value', ''),
                'Recipient_State': record_dict.get('recipient_state', '')
            },
            'anomaly_score': prediction.get('anomaly_score', 0),
            'is_anomaly': bool(prediction.get('is_anomaly', 0)),
            'confidence': prediction.get('confidence', 0)
        }
        results.append(result)
    
    return results


def lambda_handler(event, context):
    """Main Lambda handler"""
    start_time = time.time()
    
    try:
        # Parse request
        body = json.loads(event.get('body', '{}'))
        record_count = body.get('record_count', 10000)
        
        logger.info(f"Processing request for {record_count} records")
        
        # Validate environment variables
        if not ATHENA_OUTPUT_BUCKET:
            raise ValueError("ATHENA_OUTPUT_BUCKET environment variable not set")
        if not SAGEMAKER_ENDPOINT:
            raise ValueError("SAGEMAKER_ENDPOINT environment variable not set")
        if not FEATURE_STORE_TABLE:
            raise ValueError("FEATURE_STORE_TABLE environment variable not set")
        
        # Build Athena query to get features from feature store
        # Note: Feature store only has 8 of the 16 features. Missing features will be filled with zeros.
        # Available: total_amount, payment_year, hist_pay_avg/count/std/max, amt_to_max_ratio, is_high_risk_nature
        # Missing: number_of_payments, payment_month/quarter/dayofweek, is_weekend, hist_pay_total, amt_to_avg_ratio, is_new_recipient
        # Features will be sent to SageMaker as list of dicts with feature names as keys
        feature_columns = ', '.join(FEATURE_STORE_FEATURES)
        query = f"""
        SELECT 
            covered_recipient_profile_id,
            {feature_columns}
        FROM {FEATURE_STORE_DATABASE}.{FEATURE_STORE_TABLE}
        WHERE total_amount_of_payment_usdollars IS NOT NULL
        ORDER BY RAND()
        LIMIT {record_count}
        """
        
        # Query Athena feature store
        logger.info(f"Querying feature store: {FEATURE_STORE_DATABASE}.{FEATURE_STORE_TABLE}")
        columns, records = query_athena(query, FEATURE_STORE_DATABASE)
        logger.info(f"Retrieved {len(records)} records from feature store")
        
        if not records:
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': False,
                    'message': 'No records found in Athena'
                })
            }
        
        # Prepare features
        logger.info("Preparing features...")
        features = prepare_features(records, columns)
        
        # Call SageMaker endpoint
        logger.info("Calling SageMaker endpoint...")
        predictions = call_sagemaker_endpoint(features)
        
        # Handle prediction format - check if wrapped or direct list
        if isinstance(predictions, dict):
            prediction_list = predictions.get('predictions', predictions)
        elif isinstance(predictions, list):
            prediction_list = predictions
        else:
            prediction_list = []
        
        # Process results
        logger.info("Processing results...")
        results = process_inference_results(records, columns, prediction_list)
        
        # Calculate statistics
        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        total_records = len(results)
        anomaly_percentage = (anomaly_count / total_records * 100) if total_records > 0 else 0
        
        execution_time = time.time() - start_time
        
        response_body = {
            'success': True,
            'total_records': total_records,
            'anomaly_count': anomaly_count,
            'anomaly_percentage': anomaly_percentage,
            'results': results,
            'execution_time': execution_time,
            'timestamp': datetime.utcnow().isoformat(),
            'athena_query': query.strip()
        }
        
        logger.info(f"Request completed in {execution_time:.2f}s")
        logger.info(f"Found {anomaly_count} anomalies out of {total_records} records")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(response_body, cls=DecimalEncoder)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'message': 'Internal server error'
            })
        }
