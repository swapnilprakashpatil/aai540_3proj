export interface PaymentRecord {
  Profile_ID: string;
  Total_Amount_of_Payment_USDollars: number;
  Payment_Year: number;
  Nature_of_Payment: string;
  Recipient_State: string;
  // Legacy fields for backwards compatibility
  Record_ID?: string;
  Date_of_Payment?: string;
  Nature_of_Payment_or_Transfer_of_Value?: string;
  Covered_Recipient_Type?: string;
  Form_of_Payment_or_Transfer_of_Value?: string;
  Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name?: string;
  [key: string]: any;
}

export interface AnomalyResult {
  record: PaymentRecord;
  anomaly_score: number;
  is_anomaly: boolean;
  confidence: number;
  features?: any;
}

export interface AnomalyResponse {
  success: boolean;
  total_records: number;
  anomaly_count: number;
  anomaly_percentage: number;
  results: AnomalyResult[];
  execution_time: number;
  timestamp: string;
}

export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

export interface ChartDataset {
  label: string;
  data: number[];
  backgroundColor?: string | string[];
  borderColor?: string | string[];
  borderWidth?: number;
}

export interface AnomalyStats {
  totalRecords: number;
  anomalyCount: number;
  normalCount: number;
  anomalyPercentage: number;
  avgPaymentAmount: number;
  avgAnomalyAmount: number;
  avgNormalAmount: number;
}
