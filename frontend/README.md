# CMS Anomaly Detection - Angular Frontend

This is the frontend application for the CMS Open Payments Anomaly Detection system.

## Prerequisites

- Node.js (v18 or later)
- npm or yarn
- Angular CLI (`npm install -g @angular/cli`)

## Installation

```bash
npm install
```

## Development Server

```bash
npm start
```

Navigate to `http://localhost:4200/`. The application will automatically reload if you change any of the source files.

## Build

```bash
npm run build
```

Build artifacts will be stored in the `dist/` directory.

## Production Build

```bash
npm run build:prod
```

This creates an optimized production build ready for deployment.

## Environment Configuration

Update the API URL in:

- `src/environments/environment.ts` (development)
- `src/environments/environment.prod.ts` (production)

Replace `YOUR_API_GATEWAY_URL` with your actual API Gateway endpoint URL.

## Features

- **Real-time Anomaly Detection**: Analyze 10,000 random CMS payment records
- **Interactive Dashboard**: View statistics and insights
- **Multiple Chart Types**:
  - Pie chart for anomaly distribution
  - Bar chart for state-level analysis
  - Line chart for payment amount distribution
  - Scatter plot for anomaly score vs amount
- **Detailed Table View**: Filter and sort anomaly records
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

- **Angular 17**: Standalone components
- **Angular Material**: UI components
- **Chart.js + ng2-charts**: Data visualization
- **RxJS**: Reactive programming
- **TypeScript**: Type-safe development

## Deployment

This application is designed to be deployed on AWS Amplify using Terraform.
See the `terraform/` directory for infrastructure as code.
