# Statistics methodology

Reports require at least two declared seeds and reject missing, duplicate, or
extra seed results for every optimizer. The JSON aggregate retains full
floating-point precision. Display tables round only when rendered.

For each metric, the report records sample mean, sample standard deviation,
minimum, maximum, and a two-sided 95% confidence interval using Student’s
t-distribution. Optimizers sharing identical seeds are compared with a paired
t-test, Wilcoxon signed-rank test, and paired Cohen’s d. The test name and raw
p-value remain in `aggregate.json`.

