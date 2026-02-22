import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { AnomalyDetectionService } from './services/anomaly-detection.service';
import { AnomalyResponse } from './models/anomaly.model';
import { DashboardComponent } from './components/dashboard/dashboard.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    MatProgressSpinnerModule,
    MatButtonModule,
    MatCardModule,
    MatIconModule,
    MatSnackBarModule,
    DashboardComponent
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'CMS Open Payments Anomaly Detection';
  loading = false;
  anomalyData?: AnomalyResponse;
  error?: string;
  currentQuery?: string;

  constructor(
    private anomalyService: AnomalyDetectionService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.loadCachedResults();
  }

  /**
   * Load cached results on initialization
   */
  loadCachedResults(): void {
    this.anomalyService.getCachedResults().subscribe({
      next: (data) => {
        this.anomalyData = data;
      },
      error: () => {
        // No cached results, that's okay
      }
    });
  }

  /**
   * Trigger anomaly detection
   */
  detectAnomalies(): void {
    this.loading = true;
    this.error = undefined;
    
    // Show the expected query pattern immediately
    const recordCount = 10000;
    this.currentQuery = `SELECT 
    covered_recipient_profile_id,
    total_amount_of_payment_usdollars,
    payment_year,
    hist_pay_avg,
    hist_pay_count,
    hist_pay_std,
    hist_pay_max,
    amt_to_max_ratio,
    is_high_risk_nature,
    nature_of_payment_or_transfer_of_value,
    recipient_state
FROM sagemaker_featurestore.cms_payments_fg_07_22_34_14_1770503654
WHERE total_amount_of_payment_usdollars IS NOT NULL
ORDER BY RAND()
LIMIT ${recordCount}`;

    this.anomalyService.detectAnomalies(recordCount).subscribe({
      next: (data) => {
        this.anomalyData = data;
        // Update with actual query from backend if available
        if (data.athena_query) {
          this.currentQuery = data.athena_query;
        }
        this.loading = false;
        this.showNotification(`Found ${data.anomaly_count} anomalies in ${data.total_records} records`);
      },
      error: (error) => {
        this.error = error.message;
        this.loading = false;
        this.currentQuery = undefined;
        this.showNotification('Error detecting anomalies. Please try again.', 'error');
      }
    });
  }

  /**
   * Scroll to results section
   */
  scrollToResults(): void {
    const element = document.getElementById('results-section');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  /**
   * Show notification
   */
  private showNotification(message: string, type: 'success' | 'error' = 'success'): void {
    this.snackBar.open(message, 'Close', {
      duration: 5000,
      horizontalPosition: 'end',
      verticalPosition: 'top',
      panelClass: type === 'error' ? 'error-snackbar' : 'success-snackbar'
    });
  }
}
