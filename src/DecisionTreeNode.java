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
		this.setFeatures(features);
		this.setLabels(labels);
		this.setSkipIndices(skipIndices);
	}
	
	public int chooseFeature(Matrix features, Matrix labels, List<Integer> skip) {
		double min_info = Double.MAX_VALUE;
		int min_idx = -1;
		for (int k = 0; k < features.cols(); k++) {
			boolean fSkip = false;
			for (int j = 0; j < skip.size(); j++) {
				if (skip.get(j) == k) {
					fSkip = true;
				}
			}
			if (fSkip) {
				continue;
			}
			double feature_info = info(features, labels, k);
			if (feature_info < min_info) {
				min_idx = k;
				min_info = feature_info;
			}
		}
		return min_idx;
	}
	
	private double info(Matrix features, Matrix labels, int feature_idx) {
		Map<Integer, Double> histo = new TreeMap<Integer, Double>();
		double score = 0;
		for (int k = 0; k < features.valueCount(feature_idx); k++) {
			int total = 0;
			for (int j = 0; j < features.rows(); j++) {
				if (features.get(j, feature_idx) == k) {
					int label = (int) labels.get(j, 0);
					//histo.get(label)++;
					total++;
				}
			}
			for (int j = 0; j < histo.size(); j++) {
				//histo[j] /= total;
			}
			double entropy = entropy(histo);
			score += (total / features.rows()) * entropy;
		}
		return score;
	}
	
	public double getEstimate(){
		return this.getLabels().mostCommonValue(0);
	}
	
	public double decide(double[] features){
		return this.getEstimate(); 
	}
	
	public boolean splitOnFeature(int splitIndex, List<Integer> skipIndices){
		this.setSplitIndex(splitIndex);
		return false;
	}
	
	private double entropy(Map<Integer, Double> histo) {
		// TODO Auto-generated method stub
		return 0;
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