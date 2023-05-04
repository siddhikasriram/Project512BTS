
---------------------------------------------------------------------------------------------------------------------------
**COMP 512 :: ADVANCED OPERATING SYSTEMS :: FINAL PROJECT REPORT README file**
---------------------------------------------------------------------------------------------------------------------------

Submitted by: SIDDHIKA SRIRAM & NIKIL SHARAN PRABAHAR BALASUBRAMANIAN

---------------------------------------------------------------------------------------------------------------------------

Federated learning environment setup in local machine

Scripting language: Python 3.9

Computation Details: CPU

-> Files submitted: 
	requirements.txt		contains all the packages needed to run the program
	client.py       		client side code
	server.py			server side code
	bts directory 			contains the segmentation deep network model code
	dataset.zip 			contains dataset required for the clients
	saved_models/UNet 		contains the saved UNet model for segmentation
	Output.png			snippet of the output
        
Installing the requirements for the systems
-> pip install -r requirements.txt
-> unzip the dataset.zip file

Setup the server side program
-> Run ./server.py
-> Enter the IP and port number in the format <IPaddress>:<portnumber>
-> Enter the minimum number of clients participating
--server starts to listen to the clients--
-> The result received from aggregating the weights in each round from the clients is displayed to console.

Setup the client side program
-> Run ./client.py
-> Enter the Server's publicIP and port number in the format <IPaddress>:<portnumber>
--starts training process once the minimum number of clients are connected to the server--
-> The hyper parameters or the training progress from every epoch is displayed to console.

----------------------------------------------------------------------------------------------------------------------------