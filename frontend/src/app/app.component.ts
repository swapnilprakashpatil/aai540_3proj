import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
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

    this.anomalyService.detectAnomalies(10000).subscribe({
      next: (data) => {
        this.anomalyData = data;
        this.loading = false;
        this.showNotification(`Found ${data.anomaly_count} anomalies in ${data.total_records} records`);
      },
      error: (error) => {
        this.error = error.message;
        this.loading = false;
        this.showNotification('Error detecting anomalies. Please try again.', 'error');
      }
    });
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
