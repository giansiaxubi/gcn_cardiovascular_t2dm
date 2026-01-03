import torch

# DEFINE THE ANOMALY SCORE FUNCTION AND LOSS FUNCTIONS:
# 
# objective function consists of two terms:
#     ->the first term(nor_loss) minimizes the volume of the hypersphere that encloses the node embeddings of normal
#         instances
#     ->the second term (AUC_loss) encourages the score of anomaly instances to be the higher than those of normal
#         instances
# 
# the anomaly score is defined for each node as the square Euclidean distance betweem the node embedding and the
#     center c


def anomaly_score(node_embedding, c):
	#anomaly score of an instance is calculated by
	#square Euclidean distance between the node embedding and the center c
	return torch.sum((node_embedding - c) ** 2, axis=1).reshape(-1,1)

def nor_loss(node_embedding_list, c):
	#normal loss is calculated by mean squared Euclidean distance of
	#the normal node embeddings to hypersphere center c
	num_node = node_embedding_list.shape[0]
	s =  torch.sum((node_embedding_list-c)**2)
	return s/num_node

def AUC_loss(anomaly_node_emb, normal_node_emb, c):
	#auc loss encourages the score of anomaly instances to be higher than those of normal instances
	num_anomaly_node = anomaly_node_emb.size()[0]
	num_normal_node = normal_node_emb.size()[0]
	anom1 = anomaly_score(anomaly_node_emb, c)
	anom2 = anomaly_score(normal_node_emb, c)
	s = 0
	for i in range(num_anomaly_node):
		s += torch.sum(torch.sigmoid(anom1[i]-anom2)).item()
	return s/ (num_anomaly_node * num_normal_node) #check div by zero

def AnomalyDetectionLoss(anomaly_node_emb, normal_node_emb, c, regularizer=1):
	Nloss = nor_loss(normal_node_emb, c)
	AUCloss = AUC_loss(anomaly_node_emb, normal_node_emb, c)
	loss = Nloss - regularizer * AUCloss
	return loss
