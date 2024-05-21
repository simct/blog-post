---
type: docs
bookToc: True
weight: 1
---
# Exploring µ-Parameterization in Large-Scale Neural Networks

## Introduction

In the field of artificial intelligence, especially in natural language processing and computer vision, large neural network models have become a cornerstone. However, the initialization and learning rates of these models are often determined through heuristic methods, varying significantly between different studies and model sizes. This inconsistency can lead to suboptimal performance, particularly when scaling up models.

The concept of **µ-Parameterization (µP)** provides a potential solution to this problem. µP offers scaling rules for model initialization and learning rates, enabling zero-shot hyperparameter transfer from small models to larger ones. This technique promises stable training and optimal hyperparameters at scale with minimal cost. Despite its potential, µP has not been widely adopted due to its complexity and the need for further empirical validation.

In this blog post, we delve into the details of µ-Parameterization, its underlying principles, and its practical applications as explored in the paper "A Large-Scale Exploration of µ-Transfer" by Lucas Dax Lingle.

![neural_network svg](https://github.com/Choo-Minhye/--Transfer-Review/assets/45410726/6910e90e-4415-4c5d-9fc1-ae89cb170762)

*Figure 1: Conceptual illustration of neural network scaling.*

## What is µ-Parameterization?

### Concept and Principles

µ-Parameterization (µP) is a set of rules for initializing neural networks and setting learning rates that allows for the seamless transfer of hyperparameters from small proxy models to larger target models. This approach is grounded in a Gaussian Process interpretation of deep neural networks, where the width of the network (number of neurons per layer) is a critical factor.

The core idea is to scale the initialization and learning rates based on the width of the network. The general formulation of µP when training with the Adam optimizer and using an i.i.d. Gaussian initialization is as follows:

1. **Initialization Variance**: Parameters are initialized with a variance that scales inversely with the width of the layer. For example, if the width of a layer is ***M***, then the initialization variance for weight matrices  ***W<sub>AQ</sub>, W<sub>AK</sub>, W<sub>AV</sub>*** is $\frac{1}{M}$.

2. **Learning Rate**: The learning rate for each parameter is scaled based on the width of the network. For instance, the learning rate for the same weight matrices ***W<sub>AQ</sub>, W<sub>AK</sub>, W<sub>AV</sub>*** is ***$\frac{\alpha P}{M}$***, where ***$\alpha$*** is the base learning rate and ***P*** is a fixed proxy model width.

These scaling rules ensure that the behavior of small and large models remains consistent, facilitating the transfer of optimal hyperparameters across different model sizes.



### Practical Implementation

In practical terms, µP involves specific scaling rules for various components of a transformer model, which is a popular architecture in NLP and computer vision. For example, the initialization variance for the weight matrices in the attention mechanism and MLP blocks is set according to the width of the model. Similarly, the learning rate is adjusted to maintain consistent training dynamics across different scales.

Below is an example of µP scaling rules for transformers:

| Parameter | Initialization Variance | Adam Learning Rate |
|-----------|--------------------------|--------------------|
| `WE`      | 1                        | α                  |
| `WAQ`     | 1/M                      | αP/M               |
| `WAK`     | 1/M                      | αP/M               |
| `WAV`     | 1/M                      | αP/M               |
| `WAO`     | 1/(HD)                   | αP/M               |
| `WFI`     | 1/M                      | αP/M               |
| `WFO`     | 1/F                      | αP/M               |
| `WU`      | 1/M²                     | αP/M               |

*Table 1: µP scaling rules for transformers.*

In this table, `M` represents the model width, `H` the number of heads, `D` the head width, `F` the hidden width of the MLP, and `α` the base learning rate.

## µ-Transfer

One of the significant advantages of µ-Parameterization is the concept of µ-Transfer. This method allows hyperparameters, such as learning rates, found optimal in small models to be transferred directly to larger models without the need for extensive re-tuning. This process is particularly beneficial for scaling models efficiently and maintaining consistent performance across different model sizes.

### Steps in µ-Transfer

1. **Training a Small Proxy Model**: Begin by training a small proxy model, which is easier and less expensive to experiment with. Perform hyperparameter tuning on this model to find the optimal learning rate and other hyperparameters. For instance, let's denote the optimal learning rate found for this small model as \( \alpha \).

2. **Scaling the Hyperparameters**: Use the scaling rules provided by µP to adapt the hyperparameters for a larger model. The key scaling rule here is that the learning rate should be adjusted based on the ratio of the widths of the large model to the small model. For example, if the small model has a width \( 128 \) and the large model has a width \( 2048 \), the scaled learning rate for the large model would be \( \frac{\alpha \cdot 2048}{128} \).

3. **Applying the Scaled Hyperparameters**: Implement these scaled hyperparameters in the larger model. This involves adjusting the initialization variance and learning rates according to the µP rules to ensure that the training dynamics remain stable and consistent.

### Benefits of µ-Transfer

- **Efficiency**: By using µ-Transfer, researchers can avoid the costly and time-consuming process of hyperparameter tuning on large models. Instead, they can perform this tuning on smaller models and scale the results.
  
- **Consistency**: µ-Transfer helps maintain consistent training dynamics across different model sizes. This consistency is crucial for achieving optimal performance as models scale up.

- **Simplicity**: The process of scaling hyperparameters using µP is straightforward once the initial tuning on the small model is complete. This simplicity can significantly reduce the complexity of managing large-scale model training.

## Ablation Experiment

### Setup & objective
The objective of these experiments is to examine how various methods that can enhance performance affect the actual transfer of learning rates from smaller width models to larger models with µP and their impact on performance in reality. While the existing µP aimed at transferring initialization and learning rates, this experiment focused on the learning rate, an important hyperparameter in large transformer models. <br/>
The experiments were implemented using Jax/Flax on TPU V3, and the optimal learning rate was determined by measuring the validation loss. Models with widths of  {{< katex >}}M {{< katex >}} =  {{< katex >}}\{128, 512, 2048\} {{< katex >}} have parameters ranging from 4.7M to 1.2B, with the depth fixed at  {{< katex >}}L {{< katex >}} = 24. The reason for focusing solely on width and not depth is that in the case of depthwise µP, only one linear layer is used per residual block, whereas transformers use at least two layers. So, in this experiment, width is the main change to control the # of parameters.<br/>


### baseline & Summary
![baseline](https://github.com/simct/test/assets/127532891/5c0e4f35-043c-4e1f-abbf-9b166b457fbb)
The baseline represents the experimental results used as a reference for performance improvement or degradation across various experimental settings. In the baseline using µP, it is confirmed that the optimal learning rate for the smallest model is also the optimal learning rate for larger models that are 4x wider (16x larger).<br/>
The experimental results can be categorized into three main groups:
1.  **Transfer O, Performance Improvement O**
-   Cases where both learning rate transfer and performance improvement is observed.
 -  Examples:
	    Zero query initialization, Multiplicative Nonlinearities, SP unembedding initialization, Multi-query attention, batch size(4x)
2. **Transfer O, Performance Improvement X**
-   Cases where learning rate transfer, but there was no improvement in performance.
-   Examples:
		Projection biases, Embedding normalization, Cosine schedule
3. **Transfer X**
-   Cases where learning rate does not transfer.
 -  Examples:
	    RMS Norm gain, Decoupled weight decay, SP attention scale


Cases where learning rate does not transfer are not separately classified with performance improvement, as performance improvement in these cases is not significant.

### Projection Biases

![projectionbiases](https://github.com/simct/test/assets/127532891/5b7d2003-bf2f-4035-9458-4e0ead6a93d5) <br/>

Adding a bias vector to the linear layer does not guarantee an improvement in model performance. In fact, experimental results showed that the performance was similar to that of the baseline and learning rate transfer across the model size and width under µP. 


### RMS Norm gain(vector & scalar) & Embedding normalization
![RMSNorm](https://github.com/simct/test/assets/127532891/34223422-3755-487e-9221-c2beaa3cfc15)<br/>
<a href="https://arxiv.org/abs/1910.07467">RMSNorm</a> is a normalization method that uses the root mean square instead of the mean and standard deviation. 
<br/> {{< katex display=true >}}\bar{a_i}=\frac{a_i}{\mathrm{RMS(a)}}g_i, \text{where } \mathrm{RMS(a)}=\sqrt{\frac{1}{n}\sum_{i=1}^na_i^2} {{< katex >}} <br/>
To re-scale the standardized summed inputs, a trainable gain {{< /katex >}}g_i{{< /katex >}} and bias {{< /katex >}}b_i{{< /katex >}} were used. The gain can be implemented in two forms: a vector gain and a scalar multiplier. Similar to projection bias, the use of a trainable gain does not guarantee performance improvement.<br/>
The results showed that transfer does not occur in any case, vector gain and scalar multiplier, and performance degradation was observed in the models with the largest width.<br/>
On the other hand, using normalized embedding with RMSNorm without a trainable gain did not improve performance, but it was observed that the learning rate transfer was successful.


### Query Initialization
![zeroquery](https://github.com/simct/test/assets/127532891/cda1bb9e-df4d-4d95-9a42-6841e87754b8)<br/>
For query projection, µP initialization typically uses a Gaussian distribution with variance {{< /katex >}}\Theta(1/M){{< /katex >}}, but zero-initialization has been proposed to facilitate transfer. Transfer occurs with zero-initialized query, and there was a slight improvement in performance compared to the baseline, traditional Gaussian initialization.

### Cosine schedule
![cosineschedule](https://github.com/simct/test/assets/127532891/66ec153d-b017-4791-8787-4dba0176ab83)<br/>
Adjusting the learning rate over iterations is an open problem with no definitive solution, with methods like power and exponential scheduling. This experiment use <a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html">cosine scheduling</a>, which is the method that periodically decreases and increases the learning rate to prevent convergence to local minima. This approach can help the model escape suboptimal points and potentially find better solutions.The baseline used a linear schedule, but switching to cosine scheduling did not negatively impact transfer. However, a slight performance degradation was observed with cosine scheduling compared to the baseline.

### Decoupled weight decay
![weightdecay](https://github.com/simct/test/assets/127532891/4d7d83cf-03a5-434f-8456-ee34a26e067e)<br/>
When optimizing hyperparameters with Adam, the <a href="https://arxiv.org/abs/1711.05101">decoupled weight decay</a> method separates the weight decay from the optimization step, allowing independent exploration of the learning rate and weight decay factor. Experimental results using decoupled weight decay showed that optimal learning rate transfer was not achieved. A large {{< /katex >}}\lambda{{< /katex >}} value was suggested as a potential cause for this issue. In this experiment, {{< /katex >}}\lambda = 0.1{{< /katex >}} was used. The smaller difference in optimal learning rates between small and large models, compared to other transfer failures, suggested that reducing {{< /katex >}}\lambda{{< /katex >}} may help resolve the transfer problem.

### Multiplicative Nonlinearities
![nonlinear](https://github.com/simct/test/assets/127532891/b7ff437a-e3be-4bcf-9df4-74e51d2fac73)<br/>
To enhance the performance of transformers, multiplicative nonlinearity activation functions such as <a href="https://arxiv.org/abs/2002.05202">SwiGLU</a> and Squared ReLU can be utilized. SwiGLU is an activation function created by combining the Swish activation function with the Gated Linear Unit (GLU) and is considered to get better performance compared to traditional activation methods. . The formulas for each are as follows: <br/><br/>
{{< /katex >}}Swish(x) = x\sigma (\beta x){{< /katex >}} ({{< /katex >}}\sigma{{< /katex >}} : sigmoid function, {{< /katex >}}\beta{{< /katex >}} : trainable parameter )<br/>
{{< /katex >}}GLU(x,W,V,b,c) = \sigma(xW+b)\otimes (xV+c){{< /katex >}} ({{< /katex >}}W,V{{< /katex >}} : trainable tensor, {{< /katex >}}b,c{{< /katex >}}: trainable tensor bias, {{< /katex >}}\otimes{{< /katex >}} : element-wise multiplication)<br/>
{{< /katex >}}SwiGLU(x,W,V,b,c, \beta) = Swish_\beta(xW+b)\otimes(xV+c){{< /katex >}}<br/><br/>

Similarly, Squared ReLU, which is obtained by squaring the ReLU activation function, is known to help improve performance. Experimental results showed that they allow µ-transfer of the learning rate across model sizes and unlike the RMSNorm gain, there was a performance improvement from the perspective of multiplicative interaction.

### Standard Parameterization
![SPpart](https://github.com/simct/test/assets/127532891/a2d6ed4a-b628-485d-8491-840cc1d3ebc9)<br/>
<a href="https://arxiv.org/abs/2203.03466">Yang et al. (2021)</a> stateed that µP models perform better than SP models, and our various experiments with SP settings confirmed that some of the µP settings offer advantages in terms of transfer and model performance.<br/>
- attention scale<br/>
Usual attention scaling is {{< /katex >}}\tau^{-1} = 1/\sqrt{D}{{< /katex >}}, while µP proposes {{< /katex >}}\tau^{-1} = \Theta(1/D){{< /katex >}}, and baseline experiment was implemented using a simple {{< /katex >}}(1/D{{< /katex >}}) scaling. In this experiment, attention scaling was {{< /katex >}}1/\sqrt{D}{{< /katex >}} to check the SP setting. For the {{< /katex >}}M = 128{{< /katex >}} model, the optimal learning rate was {{< /katex >}}2^{-8}{{< /katex >}}, but for larger models, the optimal learning rate was changed to {{< /katex >}}2^{-6}{{< /katex >}}. This means that transfer did not occur, and performance slightly deteriorated compared to the original baseline.
- unembedding initialization<br/>
The initialization of µP's unembedding matrix follows a Gaussian distribution with a variance of {{< /katex >}}\Theta(1/M^2){{< /katex >}}, while standard parametrization (SP) uses {{< /katex >}}1/M{{< /katex >}}. Experiments using the original SP method with {{< /katex >}}1/M{{< /katex >}} showed that transfer was maintained and there was a slight improvement in performance for larger models. <br/>
<p align="center"><img src="https://github.com/simct/test/assets/127532891/597b6eef-f507-4d5c-884b-710602e45245"></p><br/>
To compare the result of the SP and µP,  this experiment was implemented using SP and compared the result with baseline's. The differences between the baseline and SP included using trainable biases in linear layers, trainable gains in RMSNorm layers, attention scale {{< /katex >}}1/\sqrt{D}{{< /katex >}}, and unembedding initialization variance {{< /katex >}}1/M{{< /katex >}}. All other hyperparameters remained the same. The combined results for SP transformers showed that transfer does not occur, and the optimal loss is lower in performance compared to the baseline. 

### Lion Optimizer
![Lion](https://github.com/simct/test/assets/127532891/1998e992-833b-4d15-94b3-45594cd04931)<br/>
<a href="https://github.com/lucidrains/lion-pytorch">The Lion optimizer</a> is known for being more than twice as memory-efficient as Adam while delivering similar performance in transformers. This optimizer restricts updates to {{< /katex >}}\{-1, 1\}{{< /katex >}} for each coordinate, yielding a coordinate size of {{< /katex >}}\Theta(1){{< /katex >}} per step. Consequently, it seems suitable to use the existing {{< /katex >}}\Theta(1/M){{< /katex >}} transfer rule as it is. However, experimental results showed that the learning rate transfer was not successful, indicating the need for further research on the transfer rule.

### Multi-query attention
![multiquery](https://github.com/simct/test/assets/127532891/70a24d8d-0692-458b-8049-b866b81760b7)<br/>
Transformer LLMs can use <a href="https://arxiv.org/abs/1911.02150">multi-query attention</a> and group generalization to increase inference speed by sharing keys/values across multiple heads. Experimental results showed that these methods lead to significant performance improvements compared to other methods, and transfer also occurred effectively.

### Batch Size(4x larger, 4x smaller)
![Batchsize](https://github.com/simct/test/assets/127532891/6e6e29f0-4edd-4ccd-b650-fa5abe0b1d47)<br/>
By adjusting the batch size while keeping the number of training tokens constant, it is possible to reduce training time or determine the minimum batch size required for operation. In this case, the learning rate formula is adapted by using twice the specified value for 4x larger batch sizes and half the value for 4x smaller batch sizes. The results showed that learning rate transfer effectively in both cases, though further research is needed to determine the optimal batch size.

### Large-scale Transfer Experiement
<p align="center"><img src="https://github.com/simct/test/assets/127532891/1a114701-44f0-4444-b73e-800c5db44e40"></p><br/>
To verify if transfer is possible over a larger scale difference, experiments was implemented by reducing {{< /katex >}}L{{< /katex >}} to 12 and setting the width to {{< /katex >}}\{128, 512, 2048, 8192\}{{< /katex >}}, resulting in models with 2M, 40M, 600M, and 10B parameters(5000x).  Zero query and Squared ReLU were used, which showed good performance and did not negatively impact transfer. The results confirmed that, despite a 5000x scale difference, the learning rate transfer well.

## Related Works
To summarize the contributions of the paper and consider the future works, this section summarizes the contents of a relevant research paper and explains its works and limitations.
- <a href="https://arxiv.org/abs/2203.03466">Tensor Programs V: Tuning large neural networks via zero-shot hyperparameter transfer</a> <br/>
The performance of µ-Transfer, which reduces computation by tuning a small model and then transferring the obtained hyperparameters to a large model instead of tuning the large parameter model directly, was demonstrated experimentally. <br/>
![relate1](https://github.com/simct/test/assets/127532891/09c64922-425d-4bbc-850a-36ca097e5fb7)<br/>
The paper experimentally confirmed that when Standard Parameterization was applied to MLP and Transformer models, the hyperparameter stability was low. However, using µ Parameterization enabled stable transfer by using width-128 network and width-8192 network. <br/>
The experiement sweeped the width and depth by changing only one of the hyperparameters—learning rate, output weight multiplier, initialization standard deviation, and learning rate schedule to check if they transfer across scale dimensions. Optimal hyperparameters transfered well across scale dimensions when the minimum width, depth, batch size, sequence length, and training steps were met. However, unlike the changes in width where hyperparameter transfer occured to wider models, the transfer to deeper models was less effective. The reason is same as the main paper's fixed depth and the experiments in this paper also adjusted the scale by fixing the depth and only varying the width.<br/>
In this paper, experiments were conducted to evaluate the efficiency and performance of µ-Transfer using various models: from a 10M to 40M Transformer on IWSLT14 De-En, from a 15M to 211M Transformer on WMT14 En-De, from a 13M to 110M BERT and to 350M BERT-large. The largest model tested was the GPT-3, where the width is shrunk from 4096 to 256, resulting in a scale difference from 40M to 6.7B, representing a 168x scale difference. The total tuning cost was only 7% of total pretraining cost and the hyperparameter was stable across the scale.<br/><br/>


Contribution of the main paper : The performance of µ-Transfer, which has been proven effective with a scale difference of up to 168x, was experimentally confirmed to be reliable even with a scale difference of up to 5000x. The paper also conducted experiments not only comparing SP (Standard Parameterization) and µP (µ Parameterization) overall but also by changing specific elements such as attention scale.

## Conclusion 

The paper aim to demonstrate that the transfer properties observed in the baseline with µP can be maintained across most scenarios. It also shows that µP outperforms standard parameterization (SP) and confirms the efficacy of part-by-part transfer for elements like attention scale and unembedding initialization, thereby validating the superiority of µP. Furthermore, it shows that transfer is feasible for models ranging from 2M to 10B parameters(5000x), suggesting applicability to larger models.<br/>
However, some issues are identified where optimal learning rate transfer does not occur, or performance decline in large models. For example, trainable RMSNorm gain and decoupled weight decay does not function properly for learning rate transfer. Although transfer is observed with projection biases and cosine scheduling, there is no performance improvement or even a decline.<br/>
The nice performance of µ-Transfer demonstrated in this paper is expected to be highly beneficial in the current AI landscape, where model sizes are continually increasing. However, since it is not always feasible to tune hyperparameters for large models, further research is required.<br/><br/>
The suggested further researchs are as follows.
- Detailed explanation and analysis of each experiment, investigating the causes of transfer failures and potential solutions through the expansion of each experiment (experiments on specific SP elements, experiments with various batch sizes)
- Scaling adjustment through depth rather than width and transferability
- Transferability of other hyperparameters
  <br/><br/>
  
The significance of this paper lies in summarizing the impact of various methods on performance and transfer under µ through ablation experiments. Nonetheless, a limitation is the focus on many ablations without conducting additional experiments for more detailed results. Although results are summarized in tables, comprehensive analysis is lacking. For instance, in cases where transfer fail, the differences in optimal learning rates between small and large models were relatively minor, suggesting potential for achieving transfer with further exploration or improvement. However, no additional experiments is implemented to investigate these possibilities. Due to the structural issues of transformers, scaling adjustments through depth make transfer more challenging. However, research on architectures that enable scaling adjustments through depth, as well as width, could increase the model's flexibility. Additionally, while the current paper focuses solely on the learning rate, it is necessary to expand the scope of transfer by experimenting with other hyperparameters such as the output weight multiplier, initialization standard deviation, and learning rate schedule, as seen in related work.<br/>




## References

- Lingle, L. D. (2024). *A Large-Scale Exploration of µ-Transfer*. arXiv preprint. Retrieved from [arXiv:2404.05728](http://arxiv.org/abs/2404.05728).
- Yang, G., & Hu, E. J. (2021). *Tensor Programs V: Tuning large neural networks via zero-shot hyperparameter transfer*. Advances in Neural Information Processing Systems.
- Biao Zhang and Rico Sennrich. Root mean square layer normalization. CoRR,
abs/1910.07467, 2019. URL http://arxiv.org/abs/1910.07467.
- Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. CoRR,
abs/1711.05101, 2017. URL http://arxiv.org/abs/1711.05101.
- Noam Shazeer. GLU variants improve transformer. CoRR, abs/2002.05202, 2020. URL
https://arxiv.org/abs/2002.05202.
- Noam Shazeer. Fast transformer decoding: One write-head is all you need. CoRR,
abs/1911.02150, 2019. URL https://arxiv.org/abs/1911.02150.

