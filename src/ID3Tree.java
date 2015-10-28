import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

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
		//System.out.println("Split index: " + splitIndex);
		
		//do the split, recursively call on them
		if(node.splitOnFeature(splitIndex, nodeSkipIndices)){
			for(DecisionTreeNode n : node.getChildren()){
				rExpandTree(n);
			}
		}
		else{	//means no more split was possible, we're at a leaf node
			Map<Integer, Double> path = new TreeMap<Integer, Double>();
			for(int i = 0; i < node.getSkipIndices().size(); ++i){
				path.put(node.getSkipIndices().get(i), node.getFeatures().row(0)[i]);
			}
			String output = "Leaf node with path: ";
			for(int i : node.getSkipIndices()){
				output += "\n\tindex: " + i + ", value: " + path.get(i);
			}
			System.out.println(output);
		}
	}
	

	public double predict(double[] features){
		DecisionTreeNode temp = root;
		return root.decide(features);
	}
	@Override
	public String toString(){
		String output = "Tree with root containing: " + root.getFeatures().rows() + " instances splits on " + root.getSplitIndex();
		return output;
	}
	
	

}

