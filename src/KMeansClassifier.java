import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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
			System.out.println("At end of round: " + ++round + "\n");
			currSSE = this.calcSSE();
			System.out.println("Total SSE: " + currSSE + "\n");
			System.out.println("Silhouette Score: " + this.calcSilhouetteScore() + "\n");
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
	
	public double calcSilhouetteScore(){
		double totalScore = 0;
		int totalInstances = 0;
		for(KMeansCluster kmc : clusters){
			for(int i = 0; i < kmc.getInstances().rows(); ++i){
				double[] instance = kmc.getInstances().row(i);
				double a_i = kmc.calcAverageDissimilarity(instance);
				double b_i = Double.MAX_VALUE;
				for(KMeansCluster kmc2: clusters){
					if(kmc2.containsInstance(instance)){
						continue;
					}
					double score = kmc2.calcAverageDissimilarity(instance);
					b_i = (score < b_i) ? score : b_i;
				}
				totalScore+= (b_i - a_i) / Math.max(b_i, a_i);
				totalInstances++;
			}
		}
		
		return totalScore/totalInstances;
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
			String file = "iris.arff";
			m.loadArff(base + file);
			Matrix m_sponge = new Matrix(m, 0, 0, m.rows(), m.cols());
			Matrix m_iris = new Matrix(m, 0, 0, m.rows(), m.cols()-1);
			KMeansClassifier kmc;
			if(file.equals("sponge.arff")){
				kmc = new KMeansClassifier(m_sponge, 4);
			}
			else if(file.equals("iris.arff")){
				m_iris.shuffle(new Random());
				kmc = new KMeansClassifier(m_iris, 6);
			}
			else{
				m.shuffle(new Random());
				m.normalize();
				kmc = new KMeansClassifier(m, 3);
			}
			kmc.cluster();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
}
