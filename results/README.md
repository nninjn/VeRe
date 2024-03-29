# Full experimental results
In this repository, we show the complete experimental results.
## Backdoor removing and Correcting safety property violation

We conduct a set of experiments on two repair tasks:

* removing backdoor

* correcting safety property violation.

In removing backdoor, we compare our method with CARE, RM and AI-Lancet. Three baselines are used in correcting safety property violation, i.e. CARE, PRDNN and REASSURE. Due to REASSURE's special mechanism, we cannot directly add it to our experiments (the time cost of repairing 10,000 samples is unacceptable). Therefore, we designed a set of new experiments for it. Specifically, we apply the limited sample experiment scenarios (i.e., 500 positive samples and different numbers of negative samples) to REASSURE and VeRe respectively for comparison. 

The experimental results are shown in the table below:

* Results of repairing violation of safety properties
<div align=center>
<img src="https://github.com/nninjn/VeRe/blob/main/images/safety_no.png">
</div>
 
 * Results of REASSURE and our method
<div align=center>
<img src="https://github.com/nninjn/VeRe/blob/main/images/reassure_no.png">
</div>

* Results of backdoor removal
<div align=center>
<img src="https://github.com/nninjn/VeRe/blob/main/images/backdoor_no.png">
</div>

## Repair with limited samples
* Results of repairing violation of safety properties with limited samples
<div align=center>
<img src="https://github.com/nninjn/VeRe/blob/main/images/safety_number_no.png">
</div>
  
* Results of backdoor removal with limited samples
<div align=center>
<img src="https://github.com/nninjn/VeRe/blob/main/images/backdoor_number_no.png">
</div>

## The coupling between the two steps
To investigate how effective our repair method is compared to these methods and how coupled the repair process is with the specific localisation and vice versa, we replaced our repair step with other repair methods. Specifically, we chose particle swarm optimization method from CARE, RM’s fine-tune and optimization with regularization (NeVer 2.0) [2] as three baselines. We denoted the replaced baselines as PSO*, RM*, and Reg* respectively. We ran additional experiments on CIFAR-10 and SVHN datasets, with two number of samples available.


*Combining with other repair methods
<div align=center>
<img src="https://github.com/nninjn/VeRe/blob/main/images/combine_no.png">
</div>

## Other activation functions
To study whether our method generalizes to networks with other activations, we add a set of experiments on backdoor defense, in which the ReLU activation function in each neural network model is replaced with the
Sigmoid activation function.

* Results of other activation function:
<div align=center>
<img src="https://github.com/nninjn/VeRe/blob/main/images/other_no.png">
</div>

