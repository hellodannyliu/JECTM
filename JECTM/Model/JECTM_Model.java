package JECTM.Model;

import Common.ComUtil;
import Common.FileUtil;
import Common.MatrixUtil;
import JECTM.DataFramework.Author;
import JECTM.DataFramework.DataSet;
import JECTM.DataFramework.ModelParameters;
import JECTM.DataFramework.ModelResultOutput;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public class JECTM_Model {
	// document related parameters
	// public Document Doc;
	public int T; // no of topics
	public int userSize; // no of users
	public int vocabularySize; // vocabulary size
	public int emoLabelSize; // no of items
	public int cogLabelSize;
	// hyperparameters
	public float alpha;
	public float beta;
	public float betaB;
	public float gamma;
	public float eta;
	public float mu;
	HashMap<String, Integer> word2id;
	//itemMap
	HashMap<String, Integer> label2id;
	// model parameters
	public int niters; // number of Gibbs sampling iteration
	// Estimated/Inferenced parameters
	public float[][] theta; // user-topic distribution, U*T
	public float[] phi; // label distribution 0 or 1, 1*2
	public float[] vPhiB; // background word distribution, 1*V
	public float[][] vPhi; // topic-word distribution, T*V
	public float[][] psi; // topic-item distribution, size T x M
	public float[][] pi; // topic-sentiment distribution, size T x S
	double topic2Probability[];
	// Temporal variables while sampling
	// public boolean[][][][] sampleY; // s x U x N_u x N_w
	// public int[][][] sampleZ; // s x U x N
	public boolean y[][][]; // U x N_u x N_w
	public int Z[][]; // U x N
	public int NW[]; // 1 x V
	public int NTW[][]; // T x V, sum is: SNTW[T]
	public int NTS[][]; // T x M, sum is: SNTS[T]
	public int NTC[][]; // T x M, sum is: SNTC[T]
	public long NY[]; // 1 x 2
	public int NUT[][]; // sum U x T, sum is: SNUT[U]
	public double SNTW[];//sum Number of topic of word
	public double SNTS[];//sum Number of topic of sentiment
	public double SNUT[];//sum Number of User of Topic
	public double SNTC[];//sum Number of topic of behavior
	private boolean isZeroProtect = false;
	ArrayList<Author> author2DocumentList;
	public JECTM_Model(ModelParameters parameters, float eta) {
		this.T = parameters.getTopicNumber();
		this.eta = eta;
		this.beta = parameters.getBeta();
		this.betaB = parameters.getBetaB();
		this.gamma = parameters.getGamma();
		this.alpha = (float) (50.0 / T);
		this.mu = eta;
		this.mu = parameters.getOmiga();
	}
	public boolean init(DataSet dataSet) {
		/**
		 * initialize the model
		 */
		this.userSize = dataSet.authorList.size();
		this.vocabularySize = dataSet.word2id.size();
		this.emoLabelSize = dataSet.sentiment2id.size();
		this.cogLabelSize = dataSet.cognitive2id.size();
		this.author2DocumentList = dataSet.authorList;
		// random assign topics
		int ZSize = 0;
		Z = new int[userSize][];
		for (int i = 0; i < userSize; i++) {
			Z[i] = new int[author2DocumentList.get(i).getDocWords().length];// Add a one-dimensional array of different lengths after the z two-dimensional array
			for (int j = 0; j < Z[i].length; j++) {//The number of cycles is the length of the last half of the two-dimensional array
				Z[i][j] = (int) Math.floor(Math.random() * T);//math.floor A number representing the largest integer less than or equal to the specified number. Random Random number between 0 and 1
				ZSize++;
				if (Z[i][j] < 0)
					Z[i][j] = 0;
				if (Z[i][j] > T - 1)
					Z[i][j] = (int) (T - 1);
			}
		}
//		System.out.println("document size of Z is ：" + ZSize);
		// assign label y
		y = new boolean[userSize][][];
		for (int i = 0; i < userSize; i++) {
			y[i] = new boolean[author2DocumentList.get(i).getDocWords().length][];
			for (int j = 0; j < author2DocumentList.get(i).getDocWords().length; j++) {
				y[i][j] = new boolean[author2DocumentList.get(i).getDocWords()[j].length];
				for (int k = 0; k < author2DocumentList.get(i).getDocWords()[j].length; k++) {
					if (Math.random() > 0.5) {
						y[i][j][k] = true;
					} else {
						y[i][j][k] = false;
					}
				}
			}
		}
		this.initialMatrix();
		this.initialSenCogMatrix(author2DocumentList, Z, y);
		this.initialMatrixSum(author2DocumentList, T);
		return true;
	}

	public void initialMatrix() {
		// initial parameters NW[] NWT[][] NIT[][] NY[] NUT[][]
		NW = new int[vocabularySize];
		vPhiB = new float[vocabularySize];
		for (int i = 0; i < vocabularySize; i++) {
			NW[i] = 0;
			vPhiB[i] = 0.0f;
		}
		NTW = new int[T][];
		vPhi = new float[T][];
		for (int t = 0; t < T; t++) {
			NTW[t] = new int[vocabularySize];
			vPhi[t] = new float[vocabularySize];
			for (int i = 0; i < vocabularySize; i++) {
				NTW[t][i] = 0;
				vPhi[t][i] = 0.0f;
			}
		}
// initial NTS data
		NTS = new int[T][];
		psi = new float[T][];
		for (int t = 0; t < T; t++) {
			NTS[t] = new int[emoLabelSize];
			psi[t] = new float[emoLabelSize];
			for (int i = 0; i < emoLabelSize; i++) {
				NTS[t][i] = 0;
				psi[t][i] = 0.0f;
			}
		}
// initial NTC data
		NTC = new int[T][];
		pi = new float[T][];
		for (int t = 0; t < T; t++) {
			NTC[t] = new int[cogLabelSize];
			pi[t] = new float[cogLabelSize];
			for (int i = 0; i < cogLabelSize; i++) {
				NTC[t][i] = 0;
				pi[t][i] = 0.0f;
			}
		}

		NY = new long[2];
		phi = new float[2];
		NY[0] = 0;
		NY[1] = 0;
		phi[0] = 0.0f;
		phi[1] = 0.0f;

		NUT = new int[userSize][];
		theta = new float[userSize][T];
		for (int i = 0; i < userSize; i++) {
			NUT[i] = new int[T];
			theta[i] = new float[T];
			for (int t = 0; t < T; t++) {
				NUT[i][t] = 0;
				theta[i][t] = 0.0f;
			}
		}
	}

	public void inference(ModelParameters parameters, String outputDir) {

		int iteration = parameters.getIteration();
		int saveStep = parameters.getSaveStep();
		int saveTimes = parameters.getSaveTimes();

		if (iteration < saveStep * saveTimes) {
			System.err.println("iteration should be at least: " + saveStep
					* saveTimes);
			System.exit(0);
		}
		int no = 0;
		// sampleY = new boolean[saveTimes][][][];
		// sampleZ = new int[saveTimes][][];
		for (int i = 0; i < iteration; i++) {
			System.out.println("iteration " + i);

			if (iteration % 10 == 0) {
				if (!checkEqual(NUT, SNUT, "NUT")
						|| !checkEqual(NTW, SNTW, "NTW")
						|| !checkEqual(NTS, SNTS, "NTS")
						|| !checkEqual(NTC, SNTC, "NTC")) {
					try {
						System.err.println("Error!!!");
						saveModelRes(outputDir + "/model-error" + (i + 1));
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			}

			for (int userId = 0; userId < userSize; userId++) {
				// sample from p(z_{u,n}|c_-{u,n},w)
				for (int userDocId = 0; userDocId < this.author2DocumentList.get(userId).getDocWords().length; userDocId++) {
					SampleTopic(this.author2DocumentList.get(userId).getDocWords()[userDocId], this.author2DocumentList.get(userId)
							.getSentiment()[userDocId],this.author2DocumentList.get(userId)
							.getCognitive()[userDocId], userId, userDocId);
//					System.out.println("294 U is ："+u);
					for (int l = 0; l < this.author2DocumentList.get(userId).getDocWords()[userDocId].length; l++)
						SampleLabel(this.author2DocumentList.get(userId).getDocWords()[userDocId][l], userId, userDocId, l);
				}
			}

			if (i >= (iteration - 1 - (saveStep * (saveTimes - 1))))
				if ((iteration - i - 1) % saveStep == 0) {
					System.out.println("Saveing the model at " + (i + 1)
							+ "-th iteration");
					// outputModelRes();
					try {
						saveModelRes(outputDir + "/model-" + (i + 1));
						// sampleY[no] = y;
						// sampleZ[no] = Z;
					} catch (Exception e) {
						e.printStackTrace();
					}
					no++;
				}
			computeModelParameter();
			this.getPerplexity(vPhi,theta,word2id,this.author2DocumentList);
		}
	}

	//得到多项式分布-+
	public void computeModelParameter() {
		System.out.println("computing model parameters...");
		for (int w = 0; w < vocabularySize; w++) {
			vPhiB[w] = (float) ((NW[w] + betaB) / (NY[0] + vocabularySize * betaB));
		}
		for (int t = 0; t < T; t++) {
			for (int w = 0; w < vocabularySize; w++)
				vPhi[t][w] = (float) ((NTW[t][w] + beta) / (SNTW[t] + vocabularySize * beta));
		}
		for (int t = 0; t < T; t++) {
			for (int m = 0; m < emoLabelSize; m++) {
				psi[t][m] = (float) ((NTS[t][m] + eta) / (SNTS[t] + emoLabelSize * eta));
			}
		}
//		computing pi
		for (int t = 0; t < T; t++) {
			for (int c = 0; c < cogLabelSize; c++) {
				pi[t][c] = (float) ((NTC[t][c] + this.mu) / (SNTC[t] + cogLabelSize * this.mu));
			}
		}

		for (int i = 0; i < 2; i++) {
			phi[i] = (float) ((NY[i] + gamma) / (NY[0] + NY[1] + 2 * gamma));
		}
		for (int u = 0; u < userSize; u++) {
			for (int t = 0; t < T; t++) {
				theta[u][t] = (float) ((NUT[u][t] + alpha) / (SNUT[u] + T * alpha));
			}
		}

		double docSum = 0.0;
		for(int userId = 0;userId<this.author2DocumentList.size();userId++) {
			for (int topicId = 0; topicId < this.T; topicId++) {
				docSum = docSum + NUT[userId][topicId];
			}
		}
		this.topic2Probability = new double[this.T];
		for(int topicId = 0;topicId<this.T;topicId++){
			int topicIdSum = 0;
			for(int userId = 0;userId<this.author2DocumentList.size();userId++){
				topicIdSum = topicIdSum + NUT[userId][topicId];
			}
			this.topic2Probability[topicId] = topicIdSum / docSum;
		}

		System.out.println("model parameters are computed");
	}

	private void initialMatrixSum(ArrayList<Author> docs, int T) {
		SNUT = new double[docs.size()];
		for (int i = 0; i < docs.size(); i++) {
			SNUT[i] = MatrixUtil.sumRow(NUT, i);
		}
		SNTW = new double[T];
		SNTS = new double[T];
		SNTC = new double[T];
		for (int t = 0; t < T; t++) {
			SNTW[t] = MatrixUtil.sumRow(NTW, t);
			SNTS[t] = MatrixUtil.sumRow(NTS, t);
			SNTC[t] = MatrixUtil.sumRow(NTC, t);

		}
	}


	private void initialSenCogMatrix(ArrayList<Author> author2DocumentList, int[][] newZ,
									 boolean[][][] newY) {
		for (int i = 0; i < userSize; i++) {
			for (int j = 0; j < newZ[i].length; j++) {
				NUT[i][newZ[i][j]]++;
			}
			for (int j = 0; j < newY[i].length; j++) {
				for (int k = 0; k < newY[i][j].length; k++) {
					if (newY[i][j][k]) {
						NTW[Z[i][j]][author2DocumentList.get(i).getDocWords()[j][k]]++;
						NY[1]++;
					} else {
						NW[author2DocumentList.get(i).getDocWords()[j][k]]++;
						NY[0]++;
					}
				}
			}
			for (int j = 0; j < author2DocumentList.get(i).getSentiment().length; j++) {
				for (int k = 0; k < author2DocumentList.get(i).getSentiment()[j].length; k++)
					NTS[Z[i][j]][author2DocumentList.get(i).getSentiment()[j][k]]++;
			}
			for (int j = 0; j < author2DocumentList.get(i).getCognitive().length; j++) {
				for (int k = 0; k < author2DocumentList.get(i).getCognitive()[j].length; k++)
					NTC[Z[i][j]][author2DocumentList.get(i).getCognitive()[j][k]]++;
			}

		}
	}

	public void getResFromLastIteration(ArrayList<Author> docs) {
		System.out.println("getting results from last interation...");
		initialMatrix();
		initialSenCogMatrix(docs, Z, y);
	}

	private void outputErr(int word, int u, int n, int l) {
		// output Z
		System.out.println("word: " + word);
		System.out.println("Z[u][n]: " + u + " " + n + " " + l);
		System.out.println("toipc: " + Z[u][n]);
		if (y[u][n][l])
			System.out.println("y: 1");
		else
			System.out.println("y: 0");
		MatrixUtil.printArray(Z);
		for (int i = 0; i < userSize; i++) {
			System.out.println("user " + i + ": ");
			MatrixUtil.printArray(y[i]);
		}
		System.out.println("NW: ");
		MatrixUtil.printArray(NW);
		System.out.println("NTW: ");
		MatrixUtil.printArray(NTW);
		System.out.println("NTI: ");
		MatrixUtil.printArray(NTS);
		System.out.println("NY: ");
		MatrixUtil.printArray(NY);
		System.out.println("NUT: ");
		MatrixUtil.printArray(NUT);
	}

	private void outputErr(ArrayList<Integer> tempUniqueWords,
						   ArrayList<Integer> tempCounts, int[] words, int[] items, int u,
						   int n) {
		FileUtil.print(tempUniqueWords);
		FileUtil.print(tempCounts);
		System.out.println("words");
		MatrixUtil.printArray(words);
		System.out.println("items");
		MatrixUtil.printArray(items);
		// output Z
		System.out.println("Z[u][n]: " + u + " " + n + " topic: " + Z[u][n]);
		MatrixUtil.printArray(Z);
		for (int i = 0; i < userSize; i++) {
			System.out.println("user " + i + ": ");
			MatrixUtil.printArray(y[i]);
		}
		System.out.println("NW: ");
		MatrixUtil.printArray(NW);
		System.out.println("NTW: ");
		MatrixUtil.printArray(NTW);
		System.out.println("NTI: ");
		MatrixUtil.printArray(NTS);
		System.out.println("NY: ");
		MatrixUtil.printArray(NY);
		System.out.println("NUT: ");
		MatrixUtil.printArray(NUT);
		System.out.println("SNUT: ");
		MatrixUtil.printArray(SNUT);
	}

	// 公式2.9
	private boolean SampleTopic(int[] words, int[] senti,int[] cog, int userId, int userDocId) {
		int topic = Z[userId][userDocId];
		// get words and their count in [u,n]
		ArrayList<Integer> tempUniqueWords = new ArrayList<Integer>();//
		ArrayList<Integer> tempCounts = new ArrayList<Integer>();
		uniqeY(words, y[userId][userDocId], tempUniqueWords, tempCounts);
		// assume the current topic assignment is hidden
		// update NTW[T][W](y=1) NTI[T][M] NUT[U][T] in {u,n}
		if (NUT[userId][topic] == 0) {
			System.err.println("NUT " + NUT[userId][topic]);
			outputErr(tempUniqueWords, tempCounts, words, senti, userId, userDocId);
		}
		for (int w1 = 0; w1 < tempUniqueWords.size(); w1++) {
			if (NTW[topic][tempUniqueWords.get(w1)] < tempCounts.get(w1)) {
				System.err.println("NTW !! error!");
				outputErr(tempUniqueWords, tempCounts, words, cog, userId, userDocId);
			}
		}
		for (int k = 0; k < senti.length; k++) {
			if (NTS[topic][senti[k]] == 0) {
				System.err.println("NTS =0 !! error!");
				outputErr(tempUniqueWords, tempCounts, words, senti, userId, userDocId);
			}
		}
		for (int c = 0; c < cog.length; c++) {
			if (NTC[topic][cog[c]] == 0) {
				System.err.println("NTC =0 !! error!");
				outputErr(tempUniqueWords, tempCounts, words, cog, userId, userDocId);
			}
		}
		NUT[userId][topic]--;
		SNUT[userId]--;
		for (int w1 = 0; w1 < tempUniqueWords.size(); w1++) {
			NTW[topic][tempUniqueWords.get(w1)] -= tempCounts.get(w1);
			SNTW[topic] -= tempCounts.get(w1);
		}
		// remove sentiment
		for (int s = 0; s < senti.length; s++) {
			NTS[topic][senti[s]]--;
			SNTS[topic]--;
		}
		// remove cognitive
		for (int c = 0; c < cog.length; c++) {
			NTC[topic][cog[c]]--;
			SNTC[topic]--;
		}
		// get p(Z_{u,n} = z|Z_c, W, Y, I)
		double[] pt = new double[T];
		// double NUTsumRowU = MatrixUtil.sumRow(NUT, u);
		double NUTsumRowU = SNUT[userId];
		// checkEqual(NUTsumRowU, MatrixUtil.sumRow(NUT, u), "NUT");
		for (int topicId = 0; topicId < T; topicId++) {
			int wcount = 0;
			double p1 = (double) (NUT[userId][topicId] + alpha) / (NUTsumRowU + T * alpha);
			double p2 = 1.0E00;
//			循环词
			for (int w = 0; w < tempUniqueWords.size(); w++) {
				int tempvalue = NTW[topicId][tempUniqueWords.get(w)];
//                double sumRow = MatrixUtil.sumRow(NTW, i);
				double sumRow = SNTW[topicId];//The number of words a user has under the topic ID
//                checkEqual(sumRow, MatrixUtil.sumRow(NTW, i), "NTW");
				for (int numC = 0; numC < tempCounts.get(w); numC++) {
					double temp = ((double) (tempvalue + beta + numC) / ((double) sumRow + vocabularySize * beta + wcount));
					double tempp2 = p2;
					p2 = p2 * temp;
					wcount++;
					if (p2 == 0.0 && this.isZeroProtect) {
						p2 = 4.9E-310;
//                        System.out.println(tempp2);
					}
				}
			}

			// assume items only appear once
			double p3 = 1.0E00;
			// double sumRow = MatrixUtil.sumRow(NTI, i);
			double sumRowS = SNTS[topicId];//Learner-emotion-topic

			// checkEqual(sumRow, MatrixUtil.sumRow(NTI, i), "NTI");
			for (int s = 0; s < senti.length; s++) {
				p3 = p3
						* ((double) (NTS[topicId][senti[s]] + this.eta) / ((double) sumRowS
						+ emoLabelSize * this.eta ));
			}
			double p4 = 1.0E00;
			double sumRowC = SNTC[topicId];
			for (int c = 0; c < cog.length; c++) {
				p4 = p4
						* ((double) (NTC[topicId][cog[c]] + this.mu) / ((double) sumRowC
						+ cogLabelSize * this.mu));
			}

			pt[topicId] = p1 * p2 * p3 * p4;
			//System.out.println("P1 P2 P3 is ："+ p1 +'-'+ p2 +'-'+ p3);
		}

		// cummulate multinomial parameters
//        System.out.println("T is ：" + T);
//        for (int i = 0; i < T; i++) System.out.println("pt" + i + "is" + pt[i]);
		int sample = ComUtil.sample(pt, T);
//        System.out.println("sample is ：" + sample);
//        System.out.println("u and n ,U and N：" + u + '-' + n + ' ' + userSize);
		assert (sample >= 0 && sample < T) : "sample value error:" + sample;
		//	if(sample>=T) sample=19;
		Z[userId][userDocId] = sample;
		topic = sample;
		// update NTW[T][W](y=1) NTI[T][M] NUT[U][T] in {u,n}
		NUT[userId][topic]++;
		SNUT[userId]++;
		for (int w1 = 0; w1 < tempUniqueWords.size(); w1++) {
			NTW[topic][tempUniqueWords.get(w1)] += tempCounts.get(w1);
			SNTW[topic] += tempCounts.get(w1);
		}
		for (int k = 0; k < senti.length; k++) {
			NTS[topic][senti[k]]++;
			SNTS[topic]++;
		}
		for (int c = 0; c < cog.length; c++) {
			NTC[topic][cog[c]]++;
			SNTC[topic]++;
		}
		tempUniqueWords.clear();
		tempCounts.clear();
		return true;
	}

	// SampleLabel(docs.get(u).getDocWords()[n][l], u, n, l);
	private void SampleLabel(int word, int u, int n, int l) {
		// remove current y label
		if (y[u][n][l]) {
			if (NY[1] == 0) {
				System.err.println("NY ");
				outputErr(word, u, n, l);
			}
			if (NTW[Z[u][n]][word] == 0) {
				System.err.println("NTW error ");
				outputErr(word, u, n, l);
			}
			NY[1]--;
			NTW[Z[u][n]][word]--;
			SNTW[Z[u][n]]--; // important!!
		} else {
			if (NY[0] == 0) {
				System.err.println("NY ");
				outputErr(word, u, n, l);
			}
			if (NW[word] == 0) {
				System.err.println("NW error ");
				outputErr(word, u, n, l);
			}
			NY[0]--;
			NW[word]--;
		}

		// p(0) and p(1)
		double pt[] = new double[2];

		double p0 = (double) (NY[0] + gamma) / (NY[0] + NY[1] + 2 * gamma);
		double p2 = 1.0d;
		p2 = (double) (NW[word] + betaB) / (NY[0] + vocabularySize * betaB);
		double p3 = 1.0d;
		double sumRow = SNTW[Z[u][n]];
		p3 = (double) (NTW[Z[u][n]][word] + beta) / (sumRow + vocabularySize * beta);

		pt[0] = p0 * p2;
		pt[1] = pt[0] + (1 - p0) * p3;

		// cummulate multinomial parameters
		int sample = ComUtil.sample(pt, 2);
		assert (sample >= 0 && sample < 2) : "sample value error:" + sample;

		if (sample == 1) {
			NY[1]++;
			NTW[Z[u][n]][word]++;
			SNTW[Z[u][n]]++;
			y[u][n][l] = true;
		} else {
			NY[0]++;
			NW[word]++;
			y[u][n][l] = false;
		}
	}

	private boolean checkEqual(int[][] a, double[] b, String string) {
		for (int i = 0; i < a.length; i++) {
			double c = MatrixUtil.sumRow(a, i);
			if (c != b[i]) {
				System.out.println(string + "\t" + c + "\t" + b[i]);
				return false;
			}
		}
		return true;
	}

	public static void uniqeY(int[] words, boolean[] y2,
							  ArrayList<Integer> tempUniqueWords, ArrayList<Integer> tempCounts) {
		for (int i = 0; i < words.length; i++) {
			if (y2[i]) {
				if (tempUniqueWords.contains(words[i])) {
					int index = tempUniqueWords.indexOf(words[i]);
					tempCounts.set(index, tempCounts.get(index) + 1);
				} else {
					tempUniqueWords.add(words[i]);
					tempCounts.add(1);
				}
			}
		}
	}

	/**
	 * output JECTM.Model paramters
	 */
	public void outputModelRes() {
		// output Z
		System.out.println("Z[u][n]: ");
		MatrixUtil.printArray(Z);
		for (int i = 0; i < userSize; i++) {
			System.out.println("user " + i + ": ");
			MatrixUtil.printArray(y[i]);
		}
	}

	public void outTaggedDoc(DataSet dataset ,String outputDir) {

		ArrayList<Author> docs = dataset.authorList;

		HashMap<Integer, String> uniWordMap = dataset.id2word;
		HashMap<Integer, String> id2sentiment = dataset.id2sentiment;
		HashMap<Integer, String> id2cognitive = dataset.id2cognitive;

		ArrayList<String> datalines = new ArrayList<String>();
		for (int i = 0; i < docs.size(); i++) {
			for (int j = 0; j < docs.get(i).getDocWords().length; j++) {
				String tmpline = "Topic " + Z[i][j] + ": ";
				for (int k1 = 0; k1 < docs.get(i).getDocWords()[j].length; k1++) {
					if (y[i][j][k1])
						tmpline += uniWordMap.get(
								docs.get(i).getDocWords()[j][k1]).concat("_1 ");
					else
						tmpline += uniWordMap.get(
								docs.get(i).getDocWords()[j][k1]).concat("_0 ");
				}
				for (int k2 = 0; k2 < docs.get(i).getSentiment()[j].length; k2++) {
					tmpline += id2sentiment.get(docs.get(i).getSentiment()[j][k2])
							+ " ";
				}
				datalines.add(tmpline);
			}
			FileUtil.writeLines(outputDir + docs.get(i).authorName,
					datalines);
			datalines.clear();
		}
	}

	void saveModelRes(String string) throws Exception {
		BufferedWriter writer = null;
		writer = new BufferedWriter(new FileWriter(new File(string)));
		writer.write("Z[u][n]: \n");
		for (int i = 0; i < Z.length; i++) {
			for (int j = 0; j < Z[i].length; j++)
				writer.write(Z[i][j] + "\t");
			writer.write("\n");
		}
		for (int i = 0; i < y.length; i++) {
			writer.write("user " + i + ": \n");
			if (y[i] != null) {
				for (int j = 0; j < y[i].length; j++) {
					if (y[i][j] != null) {
						for (int k = 0; k < y[i][j].length; k++) {
							if (y[i][j][k])
								writer.write("1\t");
							else
								writer.write("0\t");
						}
					}
					writer.write("\n");
				}
			}
		}
		writer.flush();
		writer.close();
	}

	public boolean saveModel( String output, DataSet dataset) throws Exception {

		XSSFWorkbook result = new XSSFWorkbook();
		XSSFSheet sheet1 = result.createSheet("vPhiB-backGroundWords");
		ModelResultOutput.writeDatavPhiB(vPhiB, dataset.id2word, sheet1);

		XSSFSheet sheet2 = result.createSheet("SwiftPhi");
		ModelResultOutput.writeDataSwiftPhi(phi, sheet2);

		XSSFSheet sheet3 = result.createSheet("theta-author-topic");
		ModelResultOutput.writeDataTheta(theta, sheet3,dataset.author2grade, dataset.id2authorName);

		XSSFSheet sheet4 = result.createSheet("vPhi-topic-words");
		ModelResultOutput.writeDatavPhi(vPhi, dataset.id2word, sheet4);

		XSSFSheet sheet5 = result.createSheet("topic-sentiment");
		ModelResultOutput.writeDataPsi(psi, dataset.id2sentiment, sheet5);

		XSSFSheet sheet6 = result.createSheet("topic-cognitive");
		ModelResultOutput.writeDataPi(pi, dataset.id2cognitive, sheet6);

		XSSFSheet sheet7 = result.createSheet("Author-sentiment-cognitive");
		ModelResultOutput.writeDataAuthorBehavior(dataset.authorList, dataset.id2sentiment, dataset.id2cognitive,sheet7, dataset.author2grade,dataset.id2authorName);

		XSSFSheet sheet8 = result.createSheet("Topic-Probability");
		ModelResultOutput.writeDataTopic2Pro(this.topic2Probability,sheet8);

		XSSFSheet sheet9 = result.createSheet("author-doc-topic-items");
		ModelResultOutput.writeDataDocdata(Z,dataset,sheet9);

		XSSFSheet sheet10 = result.createSheet("theta-author-topic-weigted");
		ModelResultOutput.writeDataWeigtedTheta(theta, sheet10,dataset.author2grade, dataset.id2authorName,Z);

		try (OutputStream fileOut = new FileOutputStream(output + "result.xlsx")) {
			result.write(fileOut);
		}

		return true;
	}

	private boolean checkEqual(double a, double b, String string) {
		if (a != b) {
			System.out.println(string + "\t" + a + "\t" + b);
			return false;
		} else {
			return true;
		}
	}
	public double getPerplexity(float[][] vPhi, float[][] theta, HashMap<String,Integer> word2id, ArrayList<Author> authorList) {
		double count = 0;
		int N = 0;
		int userDocWordNo[][][] = new int[authorList.size()][][];
		for(int userNo = 0;userNo < authorList.size();userNo++){
			int docWordNo[][]  = new int[authorList.get(userNo).getDocWords().length][];
			for(int DocNo = 0;DocNo < authorList.get(userNo).getDocWords().length;DocNo++){
				docWordNo[DocNo] = authorList.get(userNo).getDocWords()[DocNo];
//                for(int wordNo=0;wordNo<authorList.get(userNo).getDocWords()[DocNo].length;wordNo++){
//                    [DocNo][wordNo] = authorList.get(userNo).getDocWords()[DocNo][wordNo];
//                }
				N = N + authorList.get(userNo).getDocWords()[DocNo].length;
			}
			userDocWordNo[userNo] = docWordNo;
		}

//        userDocNo[userNo][] = new int[authorList.get(userNo).getDocWords().length];
		double mul = 0.0;

//        theta  Learner-document-topic distribution
		for(int userNo = 0;userNo < userDocWordNo.length;userNo++) {
			for (int DocNo = 0; DocNo < userDocWordNo[userNo].length; DocNo++) {
//                if(authorList.get(userNo).getSentiment()[DocNo].length==0){
//                    System.out.println(userNo +"-"+DocNo);
//                     continue;
//                    }
				double sum = 0.0;
				for (int wordNo = 0; wordNo < userDocWordNo[userNo][DocNo].length; wordNo++) {
					int senId = authorList.get(userNo).getSentiment()[DocNo][0];
					int CogId = authorList.get(userNo).getCognitive()[DocNo][0];

					for (int topicNo = 0; topicNo < theta[userNo].length; topicNo++) {
						if(false == y[userNo][DocNo][wordNo]){
//                            sum = sum + vPhiB[DocNo]*psi[topicNo][itemId] *phi[0] ;
							sum = sum + theta[userNo][topicNo] * vPhiB[wordNo] * psi[topicNo][senId]* pi[topicNo][CogId]*phi[0]  ;

						}
//                        sum = sum + theta[userNo][topicNo] * vPhi[topicNo][wordNo] * psi[topicNo][itemId]*phi[1]  ;
						sum = sum + theta[userNo][topicNo] * vPhi[topicNo][wordNo] * psi[topicNo][senId]* pi[topicNo][CogId]*phi[1]  ;

					}
					mul = mul + Math.log(sum);
				}
			}
		}

		count = 0 - mul;
		double P = Math.exp(count / N);
		System.out.println("Perplexity:" + P);
		return P;
	}
}
