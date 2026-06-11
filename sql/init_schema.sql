-- Insurance Chained Prediction Pipeline — MSSQL Schema
-- Execute against InsuranceDB after container startup

IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'InsuranceDB')
BEGIN
    CREATE DATABASE InsuranceDB;
END
GO

USE InsuranceDB;
GO

-- Raw training / source data table (mirrors CSV structure)
IF OBJECT_ID(N'dbo.customers_raw', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.customers_raw (
        customer_id              NVARCHAR(32)   NOT NULL PRIMARY KEY,
        age                      INT            NOT NULL,
        gender                   NVARCHAR(10)   NOT NULL,
        bmi                      FLOAT          NOT NULL,
        children                 INT            NOT NULL,
        smoker                   NVARCHAR(5)    NOT NULL,
        region                   NVARCHAR(50)   NOT NULL,
        occupation               NVARCHAR(100)  NOT NULL,
        annual_income_usd        INT            NOT NULL,
        exercise_level           NVARCHAR(20)   NOT NULL,
        chronic_diseases         INT            NOT NULL,
        doctor_visits_per_year   INT            NOT NULL,
        hospitalizations_last_year INT          NOT NULL,
        alcohol_consumption_per_week INT        NOT NULL,
        insurance_plan           NVARCHAR(20)   NOT NULL,
        annual_medical_cost_usd  FLOAT          NOT NULL,
        blood_pressure           NVARCHAR(20)   NOT NULL,
        diabetes                 NVARCHAR(5)    NOT NULL,
        cholesterol              INT            NOT NULL,
        sleep_hours              FLOAT          NOT NULL,
        stress_level             FLOAT          NOT NULL,
        marital_status           NVARCHAR(20)   NOT NULL,
        created_at               DATETIME2      NOT NULL DEFAULT SYSUTCDATETIME()
    );
END
GO

-- Real-time inference logging table
IF OBJECT_ID(N'dbo.inference_logs', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.inference_logs (
        log_id                   BIGINT IDENTITY(1,1) PRIMARY KEY,
        request_id               UNIQUEIDENTIFIER NOT NULL DEFAULT NEWID(),
        customer_id              NVARCHAR(32)       NULL,
        request_payload          NVARCHAR(MAX)      NOT NULL,
        predicted_annual_premium FLOAT              NOT NULL,
        predicted_insurance_type NVARCHAR(20)       NOT NULL,
        classification_probabilities NVARCHAR(MAX)  NULL,
        model_version            NVARCHAR(100)      NULL,
        latency_ms               FLOAT              NOT NULL,
        created_at               DATETIME2          NOT NULL DEFAULT SYSUTCDATETIME()
    );
    CREATE INDEX IX_inference_logs_created_at ON dbo.inference_logs (created_at DESC);
END
GO

-- Drift monitoring audit table
IF OBJECT_ID(N'dbo.drift_reports', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.drift_reports (
        report_id                BIGINT IDENTITY(1,1) PRIMARY KEY,
        report_path              NVARCHAR(500)      NOT NULL,
        drift_detected           BIT                NOT NULL,
        summary_json             NVARCHAR(MAX)      NULL,
        created_at               DATETIME2          NOT NULL DEFAULT SYSUTCDATETIME()
    );
END
GO
