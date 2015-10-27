import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

public class ID3 extends SupervisedLearner{

	private ID3Tree tree;
	private Random rand;
	
	public ID3(Random rand){
		this.rand = rand;
	}
	
	public void train(Matrix features, Matrix labels) throws Exception {
		tree = new ID3Tree(features, labels);
		DecisionTreeNode root = tree.getRoot();
		tree.rExpandTree(root);
	}

	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = tree.predict(features);
	}

}
