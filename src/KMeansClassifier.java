import java.util.ArrayList;
import java.util.List;

public class KMeansClassifier{
	private Matrix features;
	private int k;
	private List<KMeansCluster> clusters;
	public KMeansClassifier(Matrix features, int k){
		this.features = features;
		this.k = k;
		//etc
		clusters = new ArrayList<KMeansCluster>();
		for(int i = 0; i < k; ++i){
			KMeansCluster kmc = new KMeansCluster(features.row(i));
			clusters.add(kmc);
		}
		for(int j = 0; j < features.rows(); ++j){
			clusters.get(0).addInstance(features.row(j));
		}
	}

	public void cluster(){
		int pointsChanged = 0;
		do{
			pointsChanged = 0;
			for(KMeansCluster kmc : clusters){
				for(double[] instance : kmc.getInstances()){
					KMeansCluster bestCluster = this.calcBestCluster(instance);
					if(!bestCluster.equals(kmc)){
						kmc.removeInstance(instance);
						bestCluster.addInstance(instance);
						pointsChanged++;
					}
				}
			}
		}while(pointsChanged > 0 && this.calcNewCentroids());
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
			kmc.calcNewCentroid();
		}
		return true;
	}

	public String toString(){
		return "";
	}
}
