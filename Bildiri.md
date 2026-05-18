# Communication-Efficient Federated Learning for Insider Threat Detection: A Trade-off Analysis of Gradient Compression Techniques

---

### I. Introduction

In the field of cybersecurity, Insider Threat Detection has become increasingly critical. While traditional security measures primarily focus on external threats, the misuse of authorized internal accounts can lead to significantly more devastating breaches. User and Entity Behavior Analytics (UEBA) systems offer a vital defense mechanism by leveraging machine learning to model normal user behaviors and detect anomalous deviations that signify insider threats. However, centralizing sensitive behavioral log data on a single server raises substantial privacy concerns (such as GDPR compliance) and introduces severe data breach risks.

Federated Learning (FL) provides a solution to this problem as a distributed machine learning paradigm that preserves privacy by keeping data on local devices. Nevertheless, the continuous synchronization of local model weights or gradients with the central server in FL systems creates a severe communication bottleneck, particularly in environments with constrained bandwidth.

This paper proposes an intrinsically privacy-preserving and communication-efficient Federated User and Entity Behavior Analytics (F-UEBA) framework for insider threat detection. The primary contribution of our work is a systematic analysis of the impact of gradient compression techniques (Quantization and Top-K Sparsification) on detection performance and communication cost, utilizing an advanced Bidirectional LSTM Autoencoder architecture on the CERT Insider Threat dataset.

### II. Related Work

The literature on Insider Threat Detection and UEBA heavily emphasizes the power of machine learning in anomaly detection. Önal et al. (2024) noted that in event-based systems, internal threats constitute a far more critical vulnerability than external ones. Similarly, Khaliq et al. (2019) argued that traditional cybersecurity products ignore the risks posed by legitimate users, highlighting the necessity of UEBA's comprehensive profiling approach. In practice, Görmez et al. (2023) performed feature extraction and anomaly detection from user sessions using centralized deep learning models (LSTM) on the CERT dataset. The methodologies proposed by Le et al. (2020, 2021) established standards for feature extraction from the CERT dataset, and a similar pipeline is utilized in this study.

Federated Learning (FL) has gained significant attention for enabling collaborative model training while preserving data privacy. The FedAvg algorithm proposed by McMahan et al. (2017) has become a fundamental benchmark for distributed optimization in FL. However, communication efficiency remains one of the greatest hurdles to FL's practical deployment. To overcome this bottleneck, Alistarh et al. (2017) introduced quantization techniques like QSGD, while Lin et al. (2018) successfully reduced model update sizes using sparsification methods such as Deep Gradient Compression.

Our work bridges these two major research streams (UEBA and communication-efficient FL). Rather than simply performing anomaly detection with an LSTM in a centralized environment, this study contributes to the literature by examining the effects of specific gradient compression techniques—namely Top-K sparsification and FP16 quantization—on anomaly detection accuracy over Non-IID data without compromising detection capabilities.

### III. Proposed Method

In this study, a communication-efficient and privacy-preserving Federated User and Entity Behavior Analytics (F-UEBA) framework was developed. The system consists of clients where local models are trained, a central federation server that aggregates the global model, and a plugin architecture that handles gradient compression.

**3.1. Core Model: Bidirectional LSTM Autoencoder**
An advanced Bidirectional LSTM Autoencoder was designed to learn the normal behavioral patterns of users. The bidirectional structure allows the model to evaluate input sequences from both past and future temporal contexts. The model compresses the 508-dimensional input into a 192-dimensional, and subsequently 64-dimensional, bottleneck for a more precise feature representation. This compression prevents the model from memorizing noise (denoising effect), ensuring it only learns the most fundamental behavioral patterns. Layer Normalization applied within the LSTM layers stabilized the local training processes.

**3.2. Anomaly Detection and Scoring**
Since the model is exclusively trained on normal data, insider threat detection relies on the Reconstruction Error. To minimize False Positives, a sophisticated Z-Score-based calculation is employed:
* **Hybrid Z-Scoring:** To prevent naturally high-activity users (e.g., System Administrators) from constantly generating anomalies, a dual-comparison is utilized based on the user's historical behavior (`Local Z-Score`) and the overall population's behavior (`Global Z-Score`). Global baselines are calculated using a 1% Trimmed Mean/Std to remove outlier noise.
* **Feature Focus:** Instead of averaging the error across all features, only the error of the 5 most deviating features (`Top 5`) in any given window is considered, maximizing the attack signal.
* **Temporal Persistence:** To filter out short-term noise, the detection score is calculated as the average of the worst 3 windows, rather than the maximum value of a single isolated window.
* **Multi-Vector Diversity Boost:** The anomaly score is multiplied by an exponential factor depending on the number of features simultaneously showing deviations, effectively highlighting coordinated threats.

**3.3. Communication Efficiency Plugins**
* **Quantization (float32 -> float16):** The transmission of model parameters by reducing them from 32-bit floating point to 16-bit; this halves the data size by reducing parameter precision.
* **Top-K Sparsification:** The process of transmitting only the top K gradients with the largest absolute values to the server.
* **Combination:** The sequential application of both quantization and sparsification techniques.

*(Figure 1: Architectural Diagram will be placed here)*

### IV. Experimental Setup

**4.1. Dataset**
The study utilizes the CERT Insider Threat r4.2 dataset provided by Carnegie Mellon SEI. Based on the feature extraction pipeline proposed by Le et al. (2020, 2021), 508 behavioral features were extracted. The temporal context was preserved by ordering the data chronologically and applying a Sliding Window technique (`window_size=14`). The limited number of true insider threat scenarios within the dataset provides a challenging testbed for the imbalanced anomaly detection problem.

**4.2. Federated Learning Configuration**
The system was configured across 50 clients (nodes) using the FedAvg algorithm on the server side via the Flower (flwr) framework. Training was conducted over 50 communication rounds, with 50% of the clients (`fraction_fit=0.5`) participating in each round. To evaluate the system's robustness under real-world conditions, both IID and Non-IID (Dirichlet distribution, alpha=0.5) data scenarios were tested.

**4.3. Experimental Configurations (Table 1)**
1. Baseline (no plugins)
2. Quantization-fp16
3. Top-K 0.1
4. Top-K 0.1 + Quantization-fp16
5. Top-K 0.05 (Boundary analysis)
6. Non-IID baseline
7. Non-IID + Top-K 0.1 + Quantization-fp16

**4.4. Metrics**
The models were evaluated based on PR-AUC (Precision-Recall Area Under Curve), Max-F1, Balanced Accuracy, Precision, Recall, Total Communication Cost (MB), and Convergence Round.

### V. Results and Discussion

*(Table 2: Main Comparison Table will be placed here - containing all configurations and metrics)*

*(Figure 2: Learning Curve Graph will be placed here - Max F1 Score across rounds)*

The experiments clearly demonstrate the trade-off between communication cost and anomaly detection performance:

1. **Quantization-fp16:** This method provided a 50% savings in communication cost while experiencing a negligible loss in model performance (F1: 0.907 vs baseline 0.907). This proves that FP16 quantization does not compromise the model's anomaly detection sensitivity.
2. **Top-K 0.1:** In this scenario, where only 10% of model updates are transmitted, communication savings reached 80%. Although the performance drop became noticeable (F1: 0.765), the detection capability remained at acceptable levels thanks to the noise-filtering nature of the Top-K method.
3. **Combination:** The simultaneous use of Top-K 0.1 and Quantization-fp16 achieved a massive 85% communication savings (dropping from 2148 MB to 321 MB) while maintaining an F1 score of 0.834.
4. **Boundary Point (Top-K 0.05):** Experiments where the sparsification rate was reduced to 5% showed that the data was insufficient for the LSTM model to update its temporal gates, resulting in the model's failure to learn (F1: 0.649, no convergence).
5. **Non-IID Scenarios:** In Non-IID scenarios where user behaviors significantly diverge, the combined compression techniques maintained communication efficiency while keeping the performance degradation at manageable levels.

**Discussion and Trade-off Analysis:** *(Figure 3: Communication Cost vs. Detection Performance Graph will be placed here)*
The similarity of results in threshold-based metrics (F1) across some methods stems from the highly restricted number of insiders in the dataset. However, subtle differences in PR-AUC metrics more clearly illustrated the impact of compression techniques on model stability. Overall, Quantization-fp16 stands out as the most reliable method, while combined methods offer the ideal trade-off for networks with strict bandwidth limitations.

### VI. Conclusion

This study systematically analyzed gradient compression techniques to resolve communication bottlenecks in Federated Learning systems used for Insider Threat Detection. Our proposed Bidirectional LSTM and Hybrid Z-Score architecture ensured high precision in anomaly detection; meanwhile, the FP16 Quantization technique halved the communication cost without any degradation in performance, proving to be the ideal solution for practical deployment. While Top-K sparsification methods offered more aggressive savings, it was identified that the model's learning capability breaks down below the 10% threshold (the boundary point).

Future work will investigate asynchronous communication mechanisms to tolerate hardware disparities among clients, efficient client selection algorithms, and the integration of differential privacy techniques into model updates.