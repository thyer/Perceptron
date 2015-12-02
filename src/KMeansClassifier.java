import java.util.ArrayList;
import java.util.List;

public class KMeansClassifier{
	private Matrix features;
	private int k;
	private List<KMeansCluster> clusters;
	public KMeansClassifier(Matrix features, int k) throws Exception{
		this.features = features;
		this.k = k;
		double[] dimensions = new double[features.cols()];
		for(int col = 0; col < features.cols(); ++col){
			dimensions[col] = features.valueCount(col);
		}
		clusters = new ArrayList<KMeansCluster>();
		for(int i = 0; i < k; ++i){
			Matrix m = new Matrix(features, 0, 0, 0, features.cols());
			KMeansCluster kmc = new KMeansCluster(features.row(i), m);
			clusters.add(kmc);
		}
		for(int j = 0; j < features.rows(); ++j){
			clusters.get(0).addInstance(features, j, 0, 1);
		}
	}

	public void cluster() throws Exception{
		int round = 0;
		double currSSE = this.calcSSE();
		double prevSSE = Double.MAX_VALUE;	
		currSSE = Double.MAX_VALUE;
		do{
			prevSSE = currSSE;
			for(KMeansCluster kmc : clusters){
				kmc.wipeInstances();
			}
			for(int row = 0; row < features.rows(); ++row){
				double[] instance = features.row(row);
				KMeansCluster bestCluster = this.calcBestCluster(instance);
				bestCluster.addInstance(features, row, 0, 1);
			}
			System.out.println("********************************************");
			System.out.println("At end of round: " + ++round + "\n" + this.toString());
			currSSE = this.calcSSE();
			System.out.println("Total SSE: " + currSSE + "\n\n");
		}while(prevSSE != currSSE && this.calcNewCentroids());
	}
	
	public KMeansCluster calcBestCluster(double[] instance){
		KMeansCluster output = null;
		double bestDistance = Double.MAX_VALUE;
		for(KMeansCluster kmc : clusters){
			double distance = kmc.getDistance(instance);
			if(distance < bestDistance){
				bestDistance = distance;
				output = kmc;
			}
		}
		return output;
	}

	public boolean calcNewCentroids(){
		for(KMeansCluster kmc : clusters){
			if(kmc.getInstances().rows() < 1){
				System.out.println("ERROR");
			}
			kmc.calcNewCentroid();
		}
		return true;
	}
	
	public double calcSSE(){
		double totalSSE = 0;
		for(KMeansCluster kmc : clusters){
			totalSSE += kmc.calcSSE();
		}
		return totalSSE;
	}
	public String toString(){
		String output = "";
		int i = 0;
		for (KMeansCluster kmc : clusters){
			i++;
			output+= "Cluster: " + i;
			double[] centroid = kmc.getCentroid();
			output+= "\n\tCentroid: [";
			for(int j = 0; j < centroid.length; ++j){
				if(centroid[j] == Matrix.MISSING){
					output+= "?";
				}
				else{
					output+= Math.round(centroid[j]*1000d)/1000d;
				}
				if(j < centroid.length - 1){
					output+= ", ";
				}
				else{
					output+= "]";
				}
			}
			output+= "\n\tTotal instances: " + kmc.getInstances().rows() + "\n";
			output+= "\n\tCluster SSE: " + kmc.calcSSE()+ "\n";
		}
		
		return output;
	}
	
	public static void main(String args[]){
		Matrix m = new Matrix();
		try {
			String base = "C:\\Users\\Trent\\Documents\\GitHub\\Perceptron\\src\\";
			String file = "laborWithID.arff";
			m.loadArff(base + file);
			Matrix m_mod = new Matrix(m, 0, 1, m.rows(), m.cols()-2);
			Matrix m_iris = new Matrix(m, 0, 0, m.rows(), m.cols()-1);
			KMeansClassifier kmc;
			if(file.equals("sponge.arff")){
				kmc = new KMeansClassifier(m_mod, 5);
			}
			else if(file.equals("iris.arff")){
				kmc = new KMeansClassifier(m_iris, 7);
			}
			else{
				kmc = new KMeansClassifier(m, 4);
			}
			kmc.cluster();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
}
