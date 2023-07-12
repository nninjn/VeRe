
## ICSE 2024

This repository contains all codes, data and models for the conference paper ICSE 2024 # 253 and # 845.

Due to capacity limitations, all codes, data and models are uploaded anonymously in:

https://drive.google.com/file/d/1fT3vN85WtI0dUS_jvTxIdPROL78iSlBm/view?usp=sharing

We will clean up the code soon and upload it and the readme to the repository.

## Full experimental results
In this repository, we show the complete results while some of them cannot be accommodated into the main paper due to page limit.
# Backdooe removing and Correcting safety property violation.

We conduct a set of experiments on two repair tasks: 1) removing backdoor and 2) correcting safety property violation. In removing backdoor, we compare our method with CARE, RM and AI-Lancet. Three baselines are used in correcting safety property violation, i.e. CARE, PRDNN and REASSURE. Due to REASSURE's special mechanism, we cannot directly add it to our experiments (the time cost of repairing 10,000 samples is unacceptable). Therefore, we designed a set of new experiments for it. Specifically, we apply the limited sample experiment scenarios (i.e., 500 positive samples and different numbers of negative samples) to REASSURE and VeRe respectively for comparison. The experimental results are shown in the table below:
![weight_histogram](/images/safety_no.png) 
![weight_histogram](/images/reassure_no.png)
![weight_histogram](/images/backdoor_no.png)

# Repair with limited samples.
![weight_histogram](/images/safety_number_no.png)
![weight_histogram](/images/backdoor_number_no.png)

# The coupling between the two steps.
![weight_histogram](/images/combine_no.png)

# Other activation functions.
![weight_histogram](/images/other_no.png)
