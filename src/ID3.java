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
		//this.repairMatrix(features);
		//this.repairMatrix(labels);
		
		tree = new ID3Tree(features, labels);
		DecisionTreeNode root = tree.getRoot();
		tree.rExpandTree(root);
	}

	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = tree.predict(features);
	}
	
	private void repairMatrix(Matrix toRepair){
		for(int i = 0; i < toRepair.rows(); ++i){
			for(int j = 0; j < toRepair.cols(); ++j){
				if(toRepair.get(i, j) == Matrix.MISSING){
					//whatever we want to do with it
				}
			}
		}
	}

}
