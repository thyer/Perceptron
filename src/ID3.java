import java.util.Map;
import java.util.TreeMap;

public class ID3 extends SupervisedLearner{

	private ID3Tree tree;
	
	public void train(Matrix features, Matrix labels) throws Exception {
		tree = new ID3Tree(features, labels);
		DecisionTreeNode root = tree.getRoot();
		tree.rExpandTree(root);
	}

	public void predict(double[] features, double[] labels) throws Exception {
		
	}

}
