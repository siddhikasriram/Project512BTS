
Brain Tumour Segmentation using Federated Learning

Introduction:
Brain tumors are among the most lethal forms of human malignancy, with cancer mortality projected to double in the coming decades. Early detection and treatment are crucial for improving patient outcomes. However, manual tumor segmentation in clinical settings is time-consuming and prone to errors. Therefore, automated brain tumor detection systems are essential for timely diagnosis and treatment. This project explores the use of federated learning, a distributed machine learning approach, for brain tumor segmentation using magnetic resonance imaging (MRI) data.

Objective:
The primary objective of this project is to develop an effective automated brain tumor detection system using federated learning. Federated learning allows multiple parties to collaboratively train a shared model without sharing raw data, ensuring data privacy and compliance with data protection regulations.

Methodology:

Federated Learning: Federated learning is a distributed machine learning approach that enables multiple clients or organizations to collaboratively train a shared model without exchanging raw data. Each participant trains the model using their own data, and only model updates (weights/parameters) are shared with a central server for aggregation.
Network Architecture: The segmentation model is based on a deep neural network (DNN) architecture known as U-Net, which consists of a contracting path and an expansive path. The network comprises 23 convolutional layers and is trained using the BraTS dataset.
Federated Learning Strategy: The Federated Aggregation Optimization algorithm is used to average the weights collected from all clients and update the global model iteratively until convergence.
Experimental Setup:

Three experiments were conducted to evaluate the brain tumor segmentation federated learning architecture under different computational and network conditions:
Federated server and two clients on one device.
Federated server and two clients on different devices.
Federated server hosted on an Amazon Web Services (AWS) EC2 Micro instance, with each client hosted on its own device.
Results:

The experiments demonstrated that the proposed federated learning system for brain tumor segmentation was effective under various computational and network conditions. The system showed improved performance and scalability while maintaining model precision and learning capabilities.
Conclusion:

The proposed federated learning model for brain tumor segmentation offers a promising solution for enhancing the identification and treatment of brain tumors. By leveraging multiple data sources and ensuring secure and confidential client participation, the system generates highly accurate diagnostic tools that improve patient outcomes and healthcare quality.
Future Works:

Future research may explore self-supervised models for brain lesion segmentation and the implementation of blockchains in federated learning environments to enhance data transparency and security.
