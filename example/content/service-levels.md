# Service Level Indicators (SLIs), SLOs, and Error Budgets

In SRE, reliability is quantified through metrics that reflect the user's experience.

## Service Level Indicator (SLI)

An SLI is a quantitative measure of some aspect of the level of service that is provided.

- **Common Examples**: Latency, availability, error rate, throughput.
- **Formula**: (Success Count / Total Count) \* 100

## Service Level Objective (SLO)

An SLO is a target value or range of values for a service level that is measured by an SLI.

- **Example**: 99.9% of user requests should have a latency of less than 200ms over a 30-day period.
- **Key Point**: SLOs should be realistic and agreed upon by both engineering and product teams.

## Error Budget

The Error Budget is the amount of downtime or errors that a service can tolerate before it violates its SLO.

- **Formula**: 100% - SLO%
- **Example**: If the SLO is 99.9%, the error budget is 0.1% for the period.
- **Purpose**: It provides a framework for making decisions about feature velocity versus stability. If the error budget is exhausted, new features are frozen until reliability is restored.

## The Relationship

- **SLIs** are the metrics you measure.
- **SLOs** are the targets you set for those metrics.
- **Error Budgets** are the management tool derived from those targets to balance risk and innovation.
