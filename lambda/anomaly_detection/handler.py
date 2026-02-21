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
ATHENA_OUTPUT_BUCKET = os.environ.get('ATHENA_OUTPUT_BUCKET')
SAGEMAKER_ENDPOINT = os.environ.get('SAGEMAKER_ENDPOINT')


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
        
        # Extract and prepare features based on model requirements
        # Adjust these based on your actual model's feature requirements
        try:
            features = {
                'Total_Amount_of_Payment_USDollars': float(record_dict.get('total_amount_of_payment_usdollars', 0)),
                'Recipient_State': record_dict.get('recipient_state', ''),
                'Covered_Recipient_Type': record_dict.get('covered_recipient_type', ''),
                'Form_of_Payment_or_Transfer_of_Value': record_dict.get('form_of_payment_or_transfer_of_value', ''),
                'Nature_of_Payment_or_Transfer_of_Value': record_dict.get('nature_of_payment_or_transfer_of_value', ''),
                'Record_ID': record_dict.get('record_id', ''),
                'Date_of_Payment': record_dict.get('date_of_payment', '')
            }
            features_list.append(features)
        except Exception as e:
            logger.warning(f"Error preparing features for record: {str(e)}")
            continue
    
    return features_list


def call_sagemaker_endpoint(features):
    """Call SageMaker endpoint for inference"""
    try:
        # Prepare payload
        payload = json.dumps({'instances': features})
        
        # Invoke endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
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
        
        # Assuming prediction contains 'anomaly_score' and 'is_anomaly'
        # Adjust based on your model's actual output format
        result = {
            'record': {
                'Record_ID': record_dict.get('record_id', ''),
                'Covered_Recipient_Type': record_dict.get('covered_recipient_type', ''),
                'Recipient_State': record_dict.get('recipient_state', ''),
                'Total_Amount_of_Payment_USDollars': float(record_dict.get('total_amount_of_payment_usdollars', 0)),
                'Date_of_Payment': record_dict.get('date_of_payment', ''),
                'Form_of_Payment_or_Transfer_of_Value': record_dict.get('form_of_payment_or_transfer_of_value', ''),
                'Nature_of_Payment_or_Transfer_of_Value': record_dict.get('nature_of_payment_or_transfer_of_value', ''),
                'Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name': 
                    record_dict.get('applicable_manufacturer_or_applicable_gpo_making_payment_name', '')
            },
            'anomaly_score': prediction.get('anomaly_score', prediction.get('score', -1)),
            'is_anomaly': prediction.get('is_anomaly', prediction.get('anomaly_score', -1) == -1),
            'features': prediction.get('features', {})
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
        
        # Build Athena query for random records
        query = f"""
        SELECT 
            record_id,
            covered_recipient_type,
            recipient_state,
            total_amount_of_payment_usdollars,
            date_of_payment,
            form_of_payment_or_transfer_of_value,
            nature_of_payment_or_transfer_of_value,
            applicable_manufacturer_or_applicable_gpo_making_payment_name,
            recipient_city,
            recipient_zip_code,
            number_of_payments_included_in_total_amount
        FROM {ATHENA_TABLE}
        WHERE total_amount_of_payment_usdollars IS NOT NULL
        ORDER BY RAND()
        LIMIT {record_count}
        """
        
        # Query Athena
        logger.info("Querying Athena...")
        columns, records = query_athena(query, ATHENA_DATABASE)
        logger.info(f"Retrieved {len(records)} records from Athena")
        
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
        
        # Process results
        logger.info("Processing results...")
        results = process_inference_results(records, columns, predictions.get('predictions', []))
        
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
            'timestamp': datetime.utcnow().isoformat()
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
