import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { AnomalyResponse } from '../models/anomaly.model';

@Injectable({
  providedIn: 'root'
})
export class AnomalyDetectionService {
  private apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  /**
   * Fetch anomaly detection results from the Lambda API
   * @param recordCount - Number of records to analyze (default: 10000)
   */
  detectAnomalies(recordCount: number = 10000): Observable<AnomalyResponse> {
    return this.http.post<AnomalyResponse>(`${this.apiUrl}/detect`, {
      record_count: recordCount
    }).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get cached results if available
   */
  getCachedResults(): Observable<AnomalyResponse> {
    return this.http.get<AnomalyResponse>(`${this.apiUrl}/results`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Handle HTTP errors
   */
  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An error occurred';
    
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Server-side error
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
    }
    
    console.error(errorMessage);
    return throwError(() => new Error(errorMessage));
  }
}
