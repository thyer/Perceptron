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
			output += "\n\tThis leaf decides: " + node.getEstimate();
			//System.out.println(output);
		}
	}
	

	public double predict(double[] features){
		DecisionTreeNode temp = root;
		return root.decide(features);
	}
	@Override
	public String toString(){
		String output = "";
		ArrayList<DecisionTreeNode> layer = new ArrayList<DecisionTreeNode>();
		layer.add(root);
		for (int i = 1; layer.size() > 0; ++i){
			output+= "\nLayer: " + i;
			ArrayList<DecisionTreeNode> nextLayer = new ArrayList<DecisionTreeNode>();
			for(DecisionTreeNode n : layer){
				output+="\n\tNode has split Index of : " + n.getSplitIndex() + " and " + n.getFeatures().rows() + " instance(s) ";
				if(n.getChildren()!=null){
					for(DecisionTreeNode d : n.getChildren()){
						nextLayer.add(d);
					}
				}
				else{
					output+= " (it's a leaf node)";
				}
			}
			layer = nextLayer;
		}
		return output;
	}

	public void pruneTree(Matrix mxValidationFeatures, Matrix mxValidationLabels) {
		ArrayList<DecisionTreeNode> notLeaves = new ArrayList<DecisionTreeNode>();
		int nodesPruned = 0;
		int depth = this.rDFSAddNode(notLeaves, root);
		int cBestGuessTotal = assessSplit(mxValidationFeatures, mxValidationLabels);
		//System.out.println("Best guess baseline: " + cBestGuessTotal);
		//System.out.println("notLeaves has a size of: " + notLeaves.size());
		for(DecisionTreeNode n : notLeaves){
			DecisionTreeNode[] tempChildren = n.getChildren();
			n.setChildren(null);
			int guessedRight = assessSplit(mxValidationFeatures, mxValidationLabels);
			if(guessedRight > cBestGuessTotal){
				//System.out.println("Guessed better than " + cBestGuessTotal + " by " + guessedRight);
				nodesPruned++;
				cBestGuessTotal = guessedRight;
			}
			else{
				n.setChildren(tempChildren);
			}
		}
		
	}
	
	public int getNodeCount(){
		ArrayList<DecisionTreeNode> nodes = new ArrayList<DecisionTreeNode>();
		int depth = this.rDFSAddNode(nodes, root);
		System.out.println("Depth : " + depth);
		return nodes.size();
	}
	
	private int assessSplit(Matrix features, Matrix labels){
		int guessesRight = 0;
		for(int i = 0; i < features.rows(); ++i){
			if(labels.get(i, 0) == root.decide(features.row(i))){
				guessesRight++;
			}
		}
		return guessesRight;
	}
	
	private int rDFSAddNode(ArrayList<DecisionTreeNode> toAdd, DecisionTreeNode n){
		if(n.getChildren()==null || n.getChildren().length < 2){
			toAdd.add(n);
			return 1;
		}
		else{
			int max = 0;
			for(DecisionTreeNode n1 : n.getChildren()){
				int t = rDFSAddNode(toAdd, n1);
				max = Math.max(t, max);
			}
			toAdd.add(n);
			return 1 + max;
		}
	}
	
	

}

