import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class DecisionTreeNode{
	private Matrix features;
	private Matrix labels;
	private DecisionTreeNode[] children;
	private int splitIndex;
	private List<Integer> skipIndices;
	
	public DecisionTreeNode(Matrix features, Matrix labels, List<Integer> skipIndices){
		//System.out.println("Constructing node with skipIndices: " + skipIndices.toString());
		this.setFeatures(features);
		this.setLabels(labels);
		this.setSkipIndices(skipIndices);
		this.setSplitIndex(-1);
	}
	
	public int chooseFeature(Matrix features, Matrix labels, List<Integer> skip) {
		double min_info = Double.MAX_VALUE;
		int min_idx = -1;
		
		//consider features not already split on
		for (int k = 0; k < features.cols(); k++) {
			//skip features already split on
			boolean fSkip = false;
			for (int j = 0; j < skip.size(); j++) {
				if (skip.get(j) == k) {
					fSkip = true;
				}
			}
			if (fSkip) {
				continue;
			}
			
			//find lowest info split for remaining features
			double feature_info = info(features, labels, k);
			//System.out.println("Info for feature " + k + ": " + feature_info);
			if (feature_info < min_info) {
				min_idx = k;
				min_info = feature_info;
			}
		}
		return min_idx;
	}
	
	public double getEstimate(){
		return this.getLabels().mostCommonValue(0);
	}
	
	private double info(Matrix features, Matrix labels, int feature_idx) {
		Map<Integer, Double> histo = new TreeMap<Integer, Double>();
		double score = 0;
		for (int k = 0; k < features.valueCount(feature_idx); k++) {
			double total = 0;
			for (int j = 0; j < features.rows(); j++) {
				if (features.get(j, feature_idx) == k) {
					int label = (int) labels.get(j, 0);
					Double value = 1.0;
					if(histo.get(label) != null){
						value += histo.get(label);
					}
					histo.put(label, value);
					total++;
				}
			}
			for (int j : histo.keySet()) {
				histo.put(j, histo.get(j)/total);
			}
			double entropy = entropy(histo);
			score += (total / features.rows()) * entropy;
		}
		//System.out.println("Histo for index: " + feature_idx + ": " + histo.toString());
		return score;
	}
	
	public double decide(double[] features){
		if(this.children == null || this.children.length < 2){
			//System.out.println("Children of deciding node is: " + this.children);
			//System.out.println("\tskip indices are: " + this.skipIndices.toString());
			return this.getEstimate();
		}
		else{
			//System.out.println("Propagating to child for decide");
			for(DecisionTreeNode n : children){
				if (n.getFeatures().row(0)[this.splitIndex] == features[this.splitIndex]){
					return n.decide(features);
				}
			}
			
		}
		//System.out.println("NO CHILDREN FOUND WITH VALUE: " + features[splitIndex] + " ON INDEX " + splitIndex);
		return this.getEstimate();
	}
	
	public boolean splitOnFeature(int splitIndex, List<Integer> skipIndices){
		if(splitIndex == -1){
			return false;
		}
		this.setSplitIndex(splitIndex);
		boolean successfulSplit = false;
		Matrix[] childrenFeatures = null;
		Matrix[] childrenLabels = null;
		
		ArrayList<DecisionTreeNode> kiddies = new ArrayList<DecisionTreeNode>();
		MatrixSplitter ms = new MatrixSplitter(this.getFeatures(), this.getLabels(), splitIndex);
		try{
			childrenFeatures = ms.getSplitFeatures();
			childrenLabels = ms.getSplitLabels();
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
		if(childrenFeatures == null || childrenLabels == null || childrenFeatures.length < 2){
			return false;
		}
		
		for(int i = 0; i < childrenFeatures.length; ++i){
			ArrayList<Integer> newSkipIndices = new ArrayList<Integer>(this.getSkipIndices());
			newSkipIndices.add(splitIndex);
			if(childrenFeatures[i].rows() > 0){
				kiddies.add(new DecisionTreeNode(childrenFeatures[i], childrenLabels[i], newSkipIndices));
				successfulSplit = true;
			}
			else{
				kiddies.add(null);	//this basically says there's no rows for this particular split, so don't go down this path
			}
		}
		
		this.setChildren(kiddies.toArray(new DecisionTreeNode[kiddies.size()]));
		if(successfulSplit){
			return true;
		}
		else{
			return false;
		}
	}
	
	private double entropy(Map<Integer, Double> histo) {
		double output = 0;
		for (int key : histo.keySet()){
			double proportion = histo.get(key);
			output += -1 * proportion * Math.log(proportion)/Math.log(2.0);
		}
		return output;
	}
	

	
	
	public void setChildren(DecisionTreeNode[] nodes){
		this.children = nodes;
	}

	public DecisionTreeNode[] getChildren() {
		return children;
	}

	public int getSplitIndex() {
		return splitIndex;
	}

	public void setSplitIndex(int splitIndex) {
		this.splitIndex = splitIndex;
	}

	public Matrix getFeatures() {
		return features;
	}

	public void setFeatures(Matrix features) {
		this.features = features;
	}

	public Matrix getLabels() {
		return labels;
	}

	public void setLabels(Matrix labels) {
		this.labels = labels;
	}

	public List<Integer> getSkipIndices() {
		return skipIndices;
	}

	public void setSkipIndices(List<Integer> skipIndices) {
		this.skipIndices = skipIndices;
	}
	
	
}