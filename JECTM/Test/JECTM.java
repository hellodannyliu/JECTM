package JECTM.Test;

import Common.FileUtil;
import JECTM.DataFramework.DataSet;
import JECTM.DataFramework.ModelParameters;

import java.io.File;
import java.util.ArrayList;

public class JECTM{
	public static void main(String args[]) throws Exception {
		String dir = System.getProperty("user.dir");//get program path
		String modelParasFilePath = dir + "\\data\\test\\modelParameters.txt";//ProgramPath + modelparasPath+ output_path
		ModelParameters parameters = new ModelParameters(modelParasFilePath);
		String dataRootDic = "..\\data\\";
		String authorDataPath = dataRootDic +"learner_achievement.xlsx";
		String stopWordPath = "stopwords.txt";
		String forumDataPath = dataRootDic + "discussion.xlsx";
		DataSet dataset = new DataSet(authorDataPath, forumDataPath, stopWordPath);
		// 2. grid search for best parameter settings
		System.out.println("had read data!" );
		ArrayList<Float> varList = parameters.getTestVarList("alpha");
		for (float eta : varList) {
			String outputDir = makeFileDir(parameters, eta);
			// todo: add omiga paras
			JCETM_Model JCETM = new JCETM_Model(parameters, eta);
			JCETM.init(dataset);
			JCETM.inference(parameters, outputDir + "/modelRes/");
//			JCETM.getResFromLastIteration(dataset.authorList);
			JCETM.computeModelParameter();
			// 3. output BLDAModel results
			System.out.println("saving the BLDAModel...");
			JCETM.saveModel(outputDir, dataset);
			System.out.println("Ouputting tagged Docs...");
			JCETM.outTaggedDoc(dataset, outputDir
					+ "/TaggedDocs/");
		}
		System.out.println("done");
	}
	private static String makeFileDir(ModelParameters parameters, float eta) {
		String outputDir = parameters.getResultSaveDir() + "Eta_" + eta + "/";
		FileUtil.mkdir(new File(outputDir));
		FileUtil.mkdir(new File(outputDir + "/TaggedDocs/"));
		FileUtil.mkdir(new File(outputDir + "/modelRes/"));
		return outputDir;
	}
}
