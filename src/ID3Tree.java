import java.util.ArrayList;
import java.util.List;

public class ID3Tree {
	private DecisionTreeNode root;
	
	public ID3Tree(Matrix features, Matrix labels){
		List<Integer> emptyList = new ArrayList<Integer>();
		root = new DecisionTreeNode(features, labels, emptyList);
	}
	
	public DecisionTreeNode getRoot(){
		return root;
	}
	
	public void rExpandTree(DecisionTreeNode node){
		//setup
		Matrix nodeFeatures = node.getFeatures();
		Matrix nodeLabels = node.getLabels();
		List<Integer> nodeSkipIndices = node.getSkipIndices();
		
		//calculate which feature to split on
		int splitIndex = node.chooseFeature(nodeFeatures, nodeLabels, nodeSkipIndices);
		
		//do the split, recursively call on them
		nodeSkipIndices.add(splitIndex);
		if(node.splitOnFeature(splitIndex, nodeSkipIndices)){
			for(DecisionTreeNode n : node.getChildren()){
				rExpandTree(n);
			}
		}
	}
	

	public double predict(double[] features){
		DecisionTreeNode temp = root;
		return root.decide(features);
	}
	
	

}

