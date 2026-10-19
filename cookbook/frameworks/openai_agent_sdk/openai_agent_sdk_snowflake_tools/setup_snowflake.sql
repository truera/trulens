-- Snowflake setup for Support Intelligence Agent example
-- Run this SQL in your Snowflake account before using the example

CREATE DATABASE IF NOT EXISTS SUPPORT_INTELLIGENCE;
CREATE SCHEMA IF NOT EXISTS SUPPORT_INTELLIGENCE.DATA;
USE DATABASE SUPPORT_INTELLIGENCE;
USE SCHEMA DATA;

CREATE OR REPLACE WAREHOUSE SUPPORT_INTELLIGENCE_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE;

-- Structured ticket metrics table (Cortex Analyst)
CREATE OR REPLACE TABLE TICKET_METRICS (
    TICKET_ID VARCHAR,
    CUSTOMER_NAME VARCHAR,
    PRIORITY VARCHAR,
    CATEGORY VARCHAR,
    STATUS VARCHAR,
    ASSIGNED_AGENT VARCHAR,
    CREATED_DATE DATE,
    RESOLVED_DATE DATE,
    FIRST_RESPONSE_HOURS FLOAT,
    RESOLUTION_HOURS FLOAT,
    CSAT_SCORE FLOAT
);

INSERT INTO TICKET_METRICS VALUES
('TKT001', 'Acme Corp', 'high', 'technical', 'resolved', 'Alex Kim', '2024-01-05', '2024-01-06', 0.5, 12.0, 4.5),
('TKT002', 'Globex Inc', 'medium', 'billing', 'resolved', 'Sam Patel', '2024-01-08', '2024-01-09', 1.0, 8.0, 5.0),
('TKT003', 'Initech', 'low', 'feature_request', 'closed', 'Jordan Lee', '2024-01-10', '2024-01-15', 2.0, 48.0, 3.0),
('TKT004', 'Umbrella Ltd', 'urgent', 'technical', 'resolved', 'Riley Chen', '2024-01-12', '2024-01-12', 0.2, 4.0, 4.0),
('TKT005', 'Acme Corp', 'high', 'account', 'resolved', 'Alex Kim', '2024-01-15', '2024-01-16', 0.8, 10.0, 4.5),
('TKT006', 'Stark Industries', 'medium', 'billing', 'resolved', 'Sam Patel', '2024-01-18', '2024-01-19', 1.5, 14.0, 3.5),
('TKT007', 'Wayne Enterprises', 'high', 'technical', 'resolved', 'Jordan Lee', '2024-01-20', '2024-01-21', 0.3, 6.0, 5.0),
('TKT008', 'Globex Inc', 'low', 'feature_request', 'open', 'Riley Chen', '2024-01-22', NULL, 3.0, NULL, NULL),
('TKT009', 'Initech', 'urgent', 'technical', 'resolved', 'Alex Kim', '2024-01-25', '2024-01-25', 0.1, 2.0, 5.0),
('TKT010', 'Umbrella Ltd', 'medium', 'account', 'pending', 'Sam Patel', '2024-02-01', NULL, 1.2, NULL, NULL),
('TKT011', 'Stark Industries', 'high', 'billing', 'resolved', 'Jordan Lee', '2024-02-05', '2024-02-06', 0.6, 9.0, 4.0),
('TKT012', 'Wayne Enterprises', 'low', 'technical', 'resolved', 'Riley Chen', '2024-02-08', '2024-02-10', 4.0, 36.0, 3.5),
('TKT013', 'Acme Corp', 'medium', 'feature_request', 'closed', 'Alex Kim', '2024-02-12', '2024-02-18', 2.5, 72.0, 3.0),
('TKT014', 'Globex Inc', 'urgent', 'technical', 'resolved', 'Sam Patel', '2024-02-15', '2024-02-15', 0.3, 3.0, 4.5),
('TKT015', 'Initech', 'high', 'billing', 'resolved', 'Jordan Lee', '2024-02-20', '2024-02-21', 0.7, 11.0, 4.0),
('TKT016', 'Umbrella Ltd', 'medium', 'account', 'resolved', 'Riley Chen', '2024-02-25', '2024-02-26', 1.0, 16.0, 3.5),
('TKT017', 'Stark Industries', 'high', 'technical', 'open', 'Alex Kim', '2024-03-01', NULL, 0.4, NULL, NULL),
('TKT018', 'Wayne Enterprises', 'urgent', 'billing', 'resolved', 'Sam Patel', '2024-03-05', '2024-03-05', 0.2, 2.5, 5.0),
('TKT019', 'Acme Corp', 'low', 'feature_request', 'pending', 'Jordan Lee', '2024-03-10', NULL, 5.0, NULL, NULL),
('TKT020', 'Globex Inc', 'medium', 'technical', 'resolved', 'Riley Chen', '2024-03-15', '2024-03-16', 1.0, 18.0, 4.0);

-- Knowledge base articles table (Cortex Search)
CREATE OR REPLACE TABLE KB_ARTICLES (
    ARTICLE_ID VARCHAR,
    TITLE VARCHAR,
    CONTENT TEXT,
    CATEGORY VARCHAR,
    LAST_UPDATED TIMESTAMP
);

INSERT INTO KB_ARTICLES VALUES
('KB001', 'How to Reset Your Password',
 'To reset your password, navigate to Settings > Security > Reset Password. Click the reset link and follow the email instructions. Passwords must be at least 12 characters with one uppercase letter, one number, and one special character. If you do not receive the reset email within 5 minutes, check your spam folder or contact support.',
 'account', '2024-01-01 10:00:00'),

('KB002', 'API Rate Limits and Quotas',
 'API rate limits are set at 1000 requests per minute for standard plans and 5000 requests per minute for enterprise plans. If you exceed the rate limit, you will receive a 429 Too Many Requests response. Implement exponential backoff in your client code. You can monitor your current usage via the API Dashboard under Settings > API > Usage.',
 'technical', '2024-01-15 09:00:00'),

('KB003', 'Configuring Single Sign-On (SSO)',
 'To configure SSO, go to Settings > Authentication > SSO Configuration. We support SAML 2.0 and OpenID Connect protocols. You will need your Identity Provider metadata URL, entity ID, and certificate. After configuration, test the SSO flow with a non-admin user before enforcing SSO for all users. Contact your account manager for enterprise SSO setup assistance.',
 'technical', '2024-02-01 14:00:00'),

('KB004', 'Updating Billing Information',
 'To update your billing information, navigate to Settings > Billing > Payment Methods. You can add or update credit cards, set up ACH direct debit for US accounts, or request invoice billing for enterprise plans. Changes to payment methods take effect on the next billing cycle. For billing disputes, contact billing@support.example.com within 30 days of the charge.',
 'billing', '2024-01-20 11:00:00'),

('KB005', 'Understanding Your SLA and Uptime Guarantee',
 'Our standard SLA guarantees 99.9% uptime measured monthly. Enterprise plans include a 99.99% uptime SLA. Scheduled maintenance windows are communicated 72 hours in advance and do not count against SLA calculations. If uptime falls below the guaranteed level, you are eligible for service credits. File SLA claims within 30 days at support.example.com/sla-claims.',
 'account', '2024-02-10 08:00:00'),

('KB006', 'Webhook Configuration Guide',
 'To configure webhooks, go to Settings > Integrations > Webhooks. Add your endpoint URL, select the events you want to subscribe to, and set a signing secret for payload verification. Webhooks support JSON payloads with HMAC-SHA256 signature headers. Failed deliveries are retried up to 5 times with exponential backoff. Monitor webhook delivery status in the Webhooks Dashboard.',
 'technical', '2024-02-15 16:00:00'),

('KB007', 'Troubleshooting Login Issues',
 'If you cannot log in, first verify your email address and password are correct. Check if your account has been locked after 5 failed attempts. Locked accounts automatically unlock after 30 minutes. Clear your browser cache and cookies, and try an incognito window. If using SSO, verify your identity provider status. For persistent issues, contact support with your account email.',
 'account', '2024-01-05 13:00:00'),

('KB008', 'Data Export and Backup Procedures',
 'You can export your data at any time via Settings > Data > Export. Supported formats include CSV, JSON, and Parquet. Full account exports are available for enterprise plans and include all historical data. Exports are processed asynchronously and you will receive an email with a download link when ready. Automated scheduled exports can be configured via the API.',
 'technical', '2024-03-01 10:00:00'),

('KB009', 'Managing Team Members and Roles',
 'To manage team members, go to Settings > Team > Members. You can invite new members by email, assign roles (Admin, Editor, Viewer), and set up team-based access controls. Admins can manage billing, settings, and all team members. Editors can create and modify content. Viewers have read-only access. Custom roles are available on enterprise plans.',
 'account', '2024-02-20 09:00:00'),

('KB010', 'Understanding Usage-Based Pricing',
 'Our pricing is based on monthly active users (MAU) and API calls. The free tier includes up to 1000 MAU and 10000 API calls per month. Standard plans start at $49/month for up to 10000 MAU. Enterprise pricing is custom and includes volume discounts, dedicated support, and SLA guarantees. Overage charges apply at published per-unit rates. View your usage at Settings > Billing > Usage.',
 'billing', '2024-03-10 15:00:00');

ALTER TABLE KB_ARTICLES SET CHANGE_TRACKING = TRUE;

CREATE OR REPLACE CORTEX SEARCH SERVICE KB_SEARCH
    ON CONTENT
    ATTRIBUTES TITLE, CATEGORY
    WAREHOUSE = SUPPORT_INTELLIGENCE_WH
    TARGET_LAG = '1 minute'
    AS (SELECT * FROM KB_ARTICLES);

CREATE OR REPLACE STAGE MODELS DIRECTORY = (ENABLE = TRUE);

-- Upload semantic_model.yaml to this stage:
-- PUT file://semantic_model.yaml @SUPPORT_INTELLIGENCE.DATA.MODELS AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- Ground truth: Cortex Search retrieval (expected chunks per query)
CREATE OR REPLACE TABLE GROUND_TRUTH_SEARCH (
    QUERY VARCHAR,
    EXPECTED_CHUNK_TEXT TEXT,
    EXPECT_SCORE FLOAT DEFAULT 1.0
);

INSERT INTO GROUND_TRUTH_SEARCH (QUERY, EXPECTED_CHUNK_TEXT, EXPECT_SCORE) VALUES
('How do I reset my password?',
 '[account] How to Reset Your Password\nTo reset your password, navigate to Settings > Security > Reset Password. Click the reset link and follow the email instructions. Passwords must be at least 12 characters with one uppercase letter, one number, and one special character. If you do not receive the reset email within 5 minutes, check your spam folder or contact support.',
 1.0),
('What are the API rate limits?',
 '[technical] API Rate Limits and Quotas\nAPI rate limits are set at 1000 requests per minute for standard plans and 5000 requests per minute for enterprise plans. If you exceed the rate limit, you will receive a 429 Too Many Requests response. Implement exponential backoff in your client code. You can monitor your current usage via the API Dashboard under Settings > API > Usage.',
 1.0),
('How do I set up SSO?',
 '[technical] Configuring Single Sign-On (SSO)\nTo configure SSO, go to Settings > Authentication > SSO Configuration. We support SAML 2.0 and OpenID Connect protocols. You will need your Identity Provider metadata URL, entity ID, and certificate. After configuration, test the SSO flow with a non-admin user before enforcing SSO for all users. Contact your account manager for enterprise SSO setup assistance.',
 1.0),
('How do I update my billing information?',
 '[billing] Updating Billing Information\nTo update your billing information, navigate to Settings > Billing > Payment Methods. You can add or update credit cards, set up ACH direct debit for US accounts, or request invoice billing for enterprise plans. Changes to payment methods take effect on the next billing cycle. For billing disputes, contact billing@support.example.com within 30 days of the charge.',
 1.0),
('I can''t log in to my account',
 '[account] Troubleshooting Login Issues\nIf you cannot log in, first verify your email address and password are correct. Check if your account has been locked after 5 failed attempts. Locked accounts automatically unlock after 30 minutes. Clear your browser cache and cookies, and try an incognito window. If using SSO, verify your identity provider status. For persistent issues, contact support with your account email.',
 1.0),
('How do I export my data?',
 '[technical] Data Export and Backup Procedures\nYou can export your data at any time via Settings > Data > Export. Supported formats include CSV, JSON, and Parquet. Full account exports are available for enterprise plans and include all historical data. Exports are processed asynchronously and you will receive an email with a download link when ready. Automated scheduled exports can be configured via the API.',
 1.0),
('How do I configure webhooks?',
 '[technical] Webhook Configuration Guide\nTo configure webhooks, go to Settings > Integrations > Webhooks. Add your endpoint URL, select the events you want to subscribe to, and set a signing secret for payload verification. Webhooks support JSON payloads with HMAC-SHA256 signature headers. Failed deliveries are retried up to 5 times with exponential backoff. Monitor webhook delivery status in the Webhooks Dashboard.',
 1.0),
('What is the pricing model?',
 '[billing] Understanding Usage-Based Pricing\nOur pricing is based on monthly active users (MAU) and API calls. The free tier includes up to 1000 MAU and 10000 API calls per month. Standard plans start at $49/month for up to 10000 MAU. Enterprise pricing is custom and includes volume discounts, dedicated support, and SLA guarantees. Overage charges apply at published per-unit rates. View your usage at Settings > Billing > Usage.',
 1.0);

-- Ground truth: Cortex Analyst golden SQL (expected SQL per query)
CREATE OR REPLACE TABLE GROUND_TRUTH_ANALYST (
    QUERY VARCHAR,
    GOLDEN_SQL TEXT
);

INSERT INTO GROUND_TRUTH_ANALYST (QUERY, GOLDEN_SQL) VALUES
-- Basic queries
('How many tickets are there by priority?',
 'SELECT PRIORITY, COUNT(TICKET_ID) AS TICKET_COUNT FROM TICKET_METRICS GROUP BY PRIORITY ORDER BY TICKET_COUNT DESC'),
('What is the average resolution time for technical tickets?',
 'SELECT AVG(RESOLUTION_HOURS) AS AVG_RESOLUTION_HOURS FROM TICKET_METRICS WHERE CATEGORY = ''technical'' AND RESOLUTION_HOURS IS NOT NULL'),
('Which agent has the highest CSAT score?',
 'SELECT ASSIGNED_AGENT, AVG(CSAT_SCORE) AS AVG_CSAT FROM TICKET_METRICS WHERE CSAT_SCORE IS NOT NULL GROUP BY ASSIGNED_AGENT ORDER BY AVG_CSAT DESC LIMIT 1'),
('How many tickets are currently open or pending?',
 'SELECT COUNT(TICKET_ID) AS COUNT FROM TICKET_METRICS WHERE STATUS IN (''open'', ''pending'')'),
('What is the average first response time for urgent tickets?',
 'SELECT AVG(FIRST_RESPONSE_HOURS) AS AVG_FIRST_RESPONSE FROM TICKET_METRICS WHERE PRIORITY = ''urgent'''),
-- Complex queries (multi-agg, window functions, self-joins, CASE expressions)
('Which agents resolve tickets the fastest on average, and what is their average CSAT score?',
 'SELECT ASSIGNED_AGENT, COUNT(TICKET_ID) AS TICKETS_RESOLVED, ROUND(AVG(RESOLUTION_HOURS), 1) AS AVG_RESOLUTION_HOURS, ROUND(AVG(CSAT_SCORE), 2) AS AVG_CSAT FROM TICKET_METRICS WHERE STATUS IN (''resolved'', ''closed'') AND RESOLUTION_HOURS IS NOT NULL GROUP BY ASSIGNED_AGENT HAVING COUNT(TICKET_ID) >= 3 ORDER BY AVG_RESOLUTION_HOURS ASC'),
('What is the month-over-month trend in ticket volume?',
 'SELECT DATE_TRUNC(''MONTH'', CREATED_DATE) AS MONTH, COUNT(TICKET_ID) AS TICKET_COUNT, COUNT(TICKET_ID) - LAG(COUNT(TICKET_ID)) OVER (ORDER BY DATE_TRUNC(''MONTH'', CREATED_DATE)) AS CHANGE_FROM_PREV_MONTH FROM TICKET_METRICS GROUP BY DATE_TRUNC(''MONTH'', CREATED_DATE) ORDER BY MONTH'),
('Show me the resolution rate and average CSAT for each ticket category.',
 'SELECT CATEGORY, COUNT(TICKET_ID) AS TOTAL_TICKETS, ROUND(COUNT(CASE WHEN STATUS IN (''resolved'', ''closed'') THEN 1 END) * 100.0 / COUNT(TICKET_ID), 1) AS RESOLUTION_RATE_PCT, ROUND(AVG(CASE WHEN CSAT_SCORE IS NOT NULL THEN CSAT_SCORE END), 2) AS AVG_CSAT FROM TICKET_METRICS GROUP BY CATEGORY ORDER BY TOTAL_TICKETS DESC'),
('Which tickets took longer to resolve than the average for their category?',
 'SELECT t.TICKET_ID, t.CATEGORY, t.ASSIGNED_AGENT, t.RESOLUTION_HOURS, ROUND(cat_avg.AVG_RES, 1) AS CATEGORY_AVG_HOURS FROM TICKET_METRICS t JOIN (SELECT CATEGORY, AVG(RESOLUTION_HOURS) AS AVG_RES FROM TICKET_METRICS WHERE RESOLUTION_HOURS IS NOT NULL GROUP BY CATEGORY) cat_avg ON t.CATEGORY = cat_avg.CATEGORY WHERE t.RESOLUTION_HOURS > cat_avg.AVG_RES ORDER BY t.RESOLUTION_HOURS DESC'),
('What percentage of tickets by priority had a first response within one hour?',
 'SELECT PRIORITY, COUNT(TICKET_ID) AS TOTAL, COUNT(CASE WHEN FIRST_RESPONSE_HOURS <= 1.0 THEN 1 END) AS MET_SLA, ROUND(COUNT(CASE WHEN FIRST_RESPONSE_HOURS <= 1.0 THEN 1 END) * 100.0 / COUNT(TICKET_ID), 1) AS SLA_MET_PCT FROM TICKET_METRICS GROUP BY PRIORITY ORDER BY SLA_MET_PCT DESC'),
('Who are the top 3 customers by ticket volume and what is their average satisfaction?',
 'SELECT CUSTOMER_NAME, COUNT(TICKET_ID) AS TICKET_COUNT, ROUND(AVG(CSAT_SCORE), 2) AS AVG_CSAT, ROUND(AVG(RESOLUTION_HOURS), 1) AS AVG_RESOLUTION_HOURS FROM TICKET_METRICS GROUP BY CUSTOMER_NAME ORDER BY TICKET_COUNT DESC LIMIT 3'),
('Show each agent''s total workload, unresolved ticket count, and how many high or urgent tickets they handle.',
 'SELECT ASSIGNED_AGENT, COUNT(TICKET_ID) AS TOTAL_ASSIGNED, COUNT(CASE WHEN STATUS IN (''open'', ''pending'') THEN 1 END) AS UNRESOLVED, COUNT(CASE WHEN PRIORITY IN (''high'', ''urgent'') THEN 1 END) AS HIGH_PRIORITY_COUNT, ROUND(AVG(FIRST_RESPONSE_HOURS), 2) AS AVG_FIRST_RESPONSE FROM TICKET_METRICS GROUP BY ASSIGNED_AGENT ORDER BY UNRESOLVED DESC, HIGH_PRIORITY_COUNT DESC'),
('What is the average days to resolution by month and priority for resolved tickets?',
 'SELECT DATE_TRUNC(''MONTH'', CREATED_DATE) AS MONTH, PRIORITY, COUNT(TICKET_ID) AS RESOLVED_COUNT, ROUND(AVG(RESOLUTION_HOURS) / 24, 1) AS AVG_DAYS_TO_RESOLVE FROM TICKET_METRICS WHERE STATUS IN (''resolved'', ''closed'') AND RESOLUTION_HOURS IS NOT NULL GROUP BY DATE_TRUNC(''MONTH'', CREATED_DATE), PRIORITY ORDER BY MONTH, PRIORITY');
