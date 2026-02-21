import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatTableModule } from '@angular/material/table';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MatSortModule } from '@angular/material/sort';
import { MatChipsModule } from '@angular/material/chips';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatTooltipModule } from '@angular/material/tooltip';
import { NgChartsModule } from 'ng2-charts';
import { ChartConfiguration, ChartData, ChartType } from 'chart.js';
import { AnomalyResponse, AnomalyResult, AnomalyStats } from '../../models/anomaly.model';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatTableModule,
    MatPaginatorModule,
    MatSortModule,
    MatChipsModule,
    MatIconModule,
    MatButtonModule,
    MatTooltipModule,
    NgChartsModule
  ],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnChanges {
  @Input() anomalyData?: AnomalyResponse;

  stats: AnomalyStats | null = null;
  anomalyRecords: AnomalyResult[] = [];
  
  // Section expand/collapse states
  infoCardsExpanded = true;
  architectureSectionExpanded = true;
  chartsExpanded = true;
  
  displayedColumns: string[] = [
    'record_id',
    'recipient_state',
    'payment_amount',
    'payment_date',
    'nature_of_payment',
    'anomaly_score',
    'status'
  ];

  // Pie Chart
  pieChartType: ChartType = 'pie';
  pieChartData: ChartData<'pie'> = {
    labels: [],
    datasets: []
  };
  pieChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#cbd5e1',
          font: { size: 12 }
        }
      },
      title: {
        display: true,
        text: 'Anomaly Distribution',
        color: '#f1f5f9',
        font: { size: 15, weight: 'bold' }
      }
    }
  };

  // Bar Chart - State Distribution
  barChartType: ChartType = 'bar';
  barChartData: ChartData<'bar'> = {
    labels: [],
    datasets: []
  };
  barChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#cbd5e1',
          font: { size: 12 }
        }
      },
      title: {
        display: true,
        text: 'Top 10 States by Anomaly Count',
        color: '#f1f5f9',
        font: { size: 15, weight: 'bold' }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: { color: '#94a3b8' },
        grid: { color: 'rgba(148, 163, 184, 0.1)' }
      },
      x: {
        ticks: { color: '#94a3b8' },
        grid: { color: 'rgba(148, 163, 184, 0.1)' }
      }
    }
  };

  // Line Chart - Payment Amount Distribution
  lineChartType: ChartType = 'line';
  lineChartData: ChartData<'line'> = {
    labels: [],
    datasets: []
  };
  lineChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#cbd5e1',
          font: { size: 12 }
        }
      },
      title: {
        display: true,
        text: 'Payment Amount Distribution (Sorted)',
        color: '#f1f5f9',
        font: { size: 15, weight: 'bold' }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: { color: '#94a3b8' },
        grid: { color: 'rgba(148, 163, 184, 0.1)' }
      },
      x: {
        ticks: { color: '#94a3b8' },
        grid: { color: 'rgba(148, 163, 184, 0.1)' }
      }
    }
  };

  // Scatter Chart - Anomaly Score vs Amount
  scatterChartType: ChartType = 'scatter';
  scatterChartData: ChartData<'scatter'> = {
    datasets: []
  };
  scatterChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#cbd5e1',
          font: { size: 12 }
        }
      },
      title: {
        display: true,
        text: 'Anomaly Score vs Payment Amount',
        color: '#f1f5f9',
        font: { size: 15, weight: 'bold' }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Payment Amount ($)',
          color: '#f1f5f9'
        },
        ticks: { color: '#94a3b8' },
        grid: { color: 'rgba(148, 163, 184, 0.1)' }
      },
      y: {
        title: {
          display: true,
          text: 'Anomaly Score',
          color: '#f1f5f9'
        },
        ticks: { color: '#94a3b8' },
        grid: { color: 'rgba(148, 163, 184, 0.1)' }
      }
    }
  };

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['anomalyData'] && this.anomalyData) {
      this.processData();
    }
  }

  toggleInfoCards(): void {
    this.infoCardsExpanded = !this.infoCardsExpanded;
  }

  toggleArchitectureSection(): void {
    this.architectureSectionExpanded = !this.architectureSectionExpanded;
  }

  toggleCharts(): void {
    this.chartsExpanded = !this.chartsExpanded;
  }

  private processData(): void {
    this.calculateStats();
    this.filterAnomalies();
    this.preparePieChart();
    this.prepareBarChart();
    this.prepareLineChart();
    this.prepareScatterChart();
  }

  private calculateStats(): void {
    if (!this.anomalyData) return;
    
    const anomalies = this.anomalyData.results.filter(r => r.is_anomaly);
    const normals = this.anomalyData.results.filter(r => !r.is_anomaly);

    const avgAnomalyAmount = anomalies.length > 0
      ? anomalies.reduce((sum, r) => sum + r.record.Total_Amount_of_Payment_USDollars, 0) / anomalies.length
      : 0;

    const avgNormalAmount = normals.length > 0
      ? normals.reduce((sum, r) => sum + r.record.Total_Amount_of_Payment_USDollars, 0) / normals.length
      : 0;

    const avgPaymentAmount = this.anomalyData.total_records > 0
      ? this.anomalyData.results.reduce((sum, r) => sum + r.record.Total_Amount_of_Payment_USDollars, 0) / 
        this.anomalyData.total_records
      : 0;

    this.stats = {
      totalRecords: this.anomalyData.total_records,
      anomalyCount: this.anomalyData.anomaly_count,
      normalCount: this.anomalyData.total_records - this.anomalyData.anomaly_count,
      anomalyPercentage: this.anomalyData.anomaly_percentage,
      avgPaymentAmount,
      avgAnomalyAmount,
      avgNormalAmount
    };
  }

  private filterAnomalies(): void {
    if (!this.anomalyData) return;
    
    this.anomalyRecords = this.anomalyData.results.filter(r => r.is_anomaly);
  }

  private preparePieChart(): void {
    if (!this.anomalyData) return;
    
    this.pieChartData = {
      labels: ['Anomalies', 'Normal'],
      datasets: [{
        data: [this.anomalyData.anomaly_count, this.anomalyData.total_records - this.anomalyData.anomaly_count],
        backgroundColor: ['#ef4444', '#14b8a6'],
        borderColor: ['#dc2626', '#0d9488'],
        borderWidth: 1
      }]
    };
  }

  private prepareBarChart(): void {
    if (!this.anomalyData) return;
    
    // Group by state and count anomalies
    const stateCount = new Map<string, number>();
    this.anomalyRecords.forEach(record => {
      const state = record.record.Recipient_State || 'Unknown';
      stateCount.set(state, (stateCount.get(state) || 0) + 1);
    });

    // Sort and take top 10
    const sortedStates = Array.from(stateCount.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    this.barChartData = {
      labels: sortedStates.map(([state]) => state),
      datasets: [{
        label: 'Anomaly Count',
        data: sortedStates.map(([, count]) => count),
        backgroundColor: '#14b8a6',
        borderColor: '#0d9488',
        borderWidth: 0,
        borderRadius: 4
      }]
    };
  }

  private prepareLineChart(): void {
    if (!this.anomalyData) return;
    
    // Sort anomalies by payment amount
    const sorted = [...this.anomalyRecords]
      .sort((a, b) => a.record.Total_Amount_of_Payment_USDollars - b.record.Total_Amount_of_Payment_USDollars);

    // Take samples for better visualization (every nth record)
    const sampleSize = Math.min(100, sorted.length);
    const step = Math.floor(sorted.length / sampleSize);
    const samples = sorted.filter((_, index) => index % (step || 1) === 0).slice(0, sampleSize);

    this.lineChartData = {
      labels: samples.map((_, index) => `${index + 1}`),
      datasets: [{
        label: 'Payment Amount ($)',
        data: samples.map(r => r.record.Total_Amount_of_Payment_USDollars),
        borderColor: '#14b8a6',
        backgroundColor: 'rgba(20, 184, 166, 0.1)',
        fill: true,
        tension: 0.4,
        borderWidth: 2
      }]
    };
  }

  private prepareScatterChart(): void {
    if (!this.anomalyData) return;
    
    const anomalyPoints = this.anomalyRecords.map(r => ({
      x: r.record.Total_Amount_of_Payment_USDollars,
      y: Math.abs(r.anomaly_score)
    }));

    const normalPoints = this.anomalyData.results
      .filter(r => !r.is_anomaly)
      .slice(0, 500) // Limit for performance
      .map(r => ({
        x: r.record.Total_Amount_of_Payment_USDollars,
        y: Math.abs(r.anomaly_score)
      }));

    this.scatterChartData = {
      datasets: [
        {
          label: 'Anomalies',
          data: anomalyPoints,
          backgroundColor: '#ef4444',
          borderColor: '#dc2626',
          pointRadius: 4,
          pointHoverRadius: 6,
          borderWidth: 0
        },
        {
          label: 'Normal',
          data: normalPoints,
          backgroundColor: '#14b8a6',
          borderColor: '#0d9488',
          pointRadius: 3,
          pointHoverRadius: 5,
          borderWidth: 0
        }
      ]
    };
  }

  formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  }

  formatDate(dateString: string): string {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleDateString('en-US');
    } catch {
      return dateString;
    }
  }
}
