# **Brain Tumor Segmentation Using Federated Learning**

## **Introduction**
Brain tumors are one of the most lethal forms of human malignancy. Recent studies estimate that cancer mortality will double in the coming decades. Computerized diagnosis of brain tumors facilitates early treatment.

Imaging techniques such as PET, CT, MRS, and MRI are routinely used to detect and diagnose malignancies. However, clinical tumor projections are still often performed manually. To reduce mortality, an effective automated brain tumor detection system is required.

## **Literature Survey**
Federated Learning (FL) technology is emerging as a solution to the limitations of centralized learning for clinical research. It allows for the distributed training of machine learning models on remote medical devices without requiring the transfer of privacy-sensitive patient data.

The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS) challenge was created for evaluating the state-of-the-art in automated brain tumor segmentation. Various automated and semi-automated segmentation algorithms have been developed to address the problem of computational tumor segmentation.

## **Methodology**

### **3.1 Network Architecture**
The structure of the neural network follows a U-net architecture, consisting of a contracting path and an expansive path. The entire network comprises 23 convolutional layers.

### **3.2 Federated Learning**
Federated learning is a distributed machine learning approach that enables multiple clients or organizations to collaboratively train a shared model without the need to exchange raw data. In federated learning, each participant trains it using their own data. The weights after the model gets trained locally are then sent to the central server hosted on the cloud for aggregation.

### **3.3 Federated Learning Clients**
The term "clients" can refer to organizations, devices, or users who collaborate in a shared machine learning model without exchanging raw patient data related to brain tumors.

### **3.4 Federated Learning Strategy**
In the first round of model training using federated learning, the central server's weights are initialized by averaging the parameters sent from all clients or edge devices in the environment. The clients send their weights to the server, which calculates the average and sends it back to them. Federated averaging (FedAvg) is a method used for distributed training involving multiple clients.

### **3.5 FL Clients and Hosting it on AWS**
Federated learning clients are the individuals or organizations that train a shared machine learning model collaboratively without exchanging raw data. Clients have their own distinct dataset and independently train their local models while maintaining data privacy.

## **Experiments**
This study tested our brain tumor segmentation federated learning architecture in three tests. The system's behavior was tested under different computational resource distribution and network setups.

### **4.1 Federated Server and Two Clients on One Device**
In the initial experiment, the federated server and two clients were installed on a single piece of hardware.

### **4.2 Federated Server and Two Clients on Different Devices**
In the second investigation, one of the clients and the federated server were on the same device, while the other client was on a separate device.

### **4.3 Federated Server with Two Client Devices**
In the third experiment, the federated server was hosted on an Amazon Web Services (AWS) EC2 Micro instance, while each client was hosted on its own device.

## **Results**
The results demonstrated that the proposed federated learning system for brain tumor segmentation was effective under a variety of computational and network conditions. The system is able to expand while retaining model precision and learning capabilities. These results demonstrate the potential for federated learning to facilitate the development of collaborative medical models while maintaining patients' confidentiality.

## **Conclusion**
Using the proposed federated learning model for segmentation, physicians could alter the identification and treatment of brain tumors. The system employs multiple data sources and secure, confidential client participation to generate highly accurate and trustworthy diagnostic tools that enhance patient outcomes and healthcare quality.

## **Future Works**
Segmentation of brain lesions may be facilitated by self-supervised models in future research in this field. It is possible to implement blockchains in federated learning environments, making it simpler to locate all of the information about each epoch from other clients.

## **References**
1. [WHO](https://www.who.int/news-room/fact-sheets/detail/cancer) (10 December 2021)
2. [Tumour Society, N.B.](https://braintumor.org/brain-tumors/about-brain-tumors/brain-tumor-facts/)
3. Bernardo Camajori Tedeschini, R.S.L.B.I.S.M.N. Stefano Savazzi, Serio
