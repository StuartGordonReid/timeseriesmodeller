/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package main;

import java.io.File;
import java.text.DecimalFormat;
import java.util.LinkedList;

import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.Greedy;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.ml.train.strategy.StopTrainingStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.util.simple.EncogUtility;

import pso.ChargedPSO;
import pso.FFNNParticle;
import pso.QuantumPSO;
import pso.StandardPSO_PE;
import pso.StandardPSO_SSE;
import recurrent.Elman;
import recurrent.Hopfield;
import recurrent.Jordan;

/**
 *
 * @author stuart
 */
public class Simulator {

    String[] datasets;
    Reporter[] results;
    LinkedList<Reporter[]> full_results;
    double errorGoal = 0.0001;
    boolean stuart = true;
    int epochs;
    DecimalFormat df = new DecimalFormat("#0.0000#");
    int samples = 55;

    Simulator(String[] dataFiles, int iterations) {
        epochs = iterations;
        datasets = dataFiles;
        results = new Reporter[datasets.length];
        for (int r = 0; r < datasets.length; r++) {
            results[r] = new Reporter(epochs, datasets[r]);
        }
    }

    public void start() {

        System.out.println("Running standard FFNN");
        simulateFFNN();

        System.out.println("Running standard Elman RNN");
        simulateElman();

        System.out.println("Running standard Jordan RNN");
        simulateJordan();

        System.out.println("Running standard PSO");
        simulateStandardPSOSSE();

        for (int i = 0; i < results.length; i++) {
            if (stuart) {
                //System.out.println("Output configured to stuart");
                switch (i) {
                    case 0:
                        results[i].printToCSV("Ouput_Brazil.csv");
                        break;
                    case 1:
                        results[i].printToCSV("Ouput_Russia.csv");
                        break;
                    case 2:
                        results[i].printToCSV("Ouput_India.csv");
                        break;
                    case 3:
                        results[i].printToCSV("Ouput_China.csv");
                        break;
                    case 4:
                        results[i].printToCSV("Ouput_SA.csv");
                        break;
                }
            } else {
                System.out.println("Output configured to simon");
            }
        }
    }

    public double getTotalPredictionError(MLDataSet dataSet, BasicNetwork nn) {
        double totalPredictionError = 0.0;
        for (int k = 0; k < dataSet.size(); k++) {
            double prediction = nn.compute(dataSet.get(k).getInput()).getData(0);
            double aim = dataSet.get(k).getIdeal().getData(0);
            if (prediction > aim) {
                double error = prediction - aim;
                totalPredictionError += error * error;
            } else {
                double error = aim - prediction;
                totalPredictionError += error * error;
            }
        }
        return totalPredictionError;
    }

    public void simulateElman() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            Elman nn = new Elman(22, 30, 2);
            CalculateScore score = new TrainingSetScore(dataSet);

            MLTrain trainAlt = new NeuralSimulatedAnnealing(nn.getParticle(), score, 10, 2, 100);
            MLTrain trainMain = new Backpropagation(nn.getParticle(), dataSet, errorGoal, 0.0);

            trainMain.addStrategy(new Greedy());
            trainMain.addStrategy(new HybridStrategy(trainAlt));

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                double value = trainMain.getError();;

                //Save the value in results
                results[i].get(j).setElmanSSE(value);

                double predictionError = getTotalPredictionError(dataSet, nn.getParticle());
                results[i].get(j).setElmanPE(predictionError);

                trainMain.iteration();
            }

            if (i == 0) {
                System.out.println("Elman end state");
                for (int k = 0; k < dataSet.size(); k++) {
                    double prediction = nn.getParticle().compute(dataSet.get(k).getInput()).getData(0);
                    double aim = dataSet.get(k).getIdeal().getData(0);
                    System.out.println(prediction + "," + aim);
                }
            }
        }
    }

    public void simulateJordan() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            Jordan nn = new Jordan(22, 30, 2);
            CalculateScore score = new TrainingSetScore(dataSet);

            MLTrain trainAlt = new NeuralSimulatedAnnealing(nn.getParticle(), score, 10, 2, 100);
            MLTrain trainMain = new Backpropagation(nn.getParticle(), dataSet, errorGoal, 0.0);

            trainMain.addStrategy(new Greedy());
            trainMain.addStrategy(new HybridStrategy(trainAlt));

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                double value = trainMain.getError();

                //Save the value in results
                results[i].get(j).setJordanSSE(value);

                double predictionError = getTotalPredictionError(dataSet, nn.getParticle());
                results[i].get(j).setJordanPE(predictionError);

                trainMain.iteration();
            }

            if (i == 0) {
                System.out.println("Jordan end state");
                for (int k = 0; k < dataSet.size(); k++) {
                    double prediction = nn.getParticle().compute(dataSet.get(k).getInput()).getData(0);
                    double aim = dataSet.get(k).getIdeal().getData(0);
                    System.out.println(prediction + "," + aim);
                }
            }
        }
    }

    public void simulateFFNN() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            FFNNParticle nn = new FFNNParticle(22, 30, 2);
            CalculateScore score = new TrainingSetScore(dataSet);

            MLTrain trainAlt = new NeuralSimulatedAnnealing(nn.getParticle(), score, 10, 2, 100);
            MLTrain trainMain = new Backpropagation(nn.getParticle(), dataSet, errorGoal, 0.0);

            trainMain.addStrategy(new Greedy());
            trainMain.addStrategy(new HybridStrategy(trainAlt));

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                double value = trainMain.getError();

                //Save the value in results
                results[i].get(j).setFFNNSSE(value);

                double predictionError = getTotalPredictionError(dataSet, nn.getParticle());
                results[i].get(j).setFFNNPE(predictionError);

                trainMain.iteration();
            }

            if (i == 0) {
                System.out.println("FFNN end state");
                for (int k = 0; k < dataSet.size(); k++) {
                    double prediction = nn.getParticle().compute(dataSet.get(k).getInput()).getData(0);
                    double aim = dataSet.get(k).getIdeal().getData(0);
                    System.out.println(prediction + "," + aim);
                }
            }
        }
    }

    public void simulateChargedPSO() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            ChargedPSO nn = new ChargedPSO();
            CalculateScore score = new TrainingSetScore(dataSet);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                double value = 0.0;

                //Save the value in results
                results[i].get(j).setChargedPSOSSE(value);
            }
        }
    }

    public void simulateQuantumPSO() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            QuantumPSO nn = new QuantumPSO();
            CalculateScore score = new TrainingSetScore(dataSet);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; i++) {

                double value = 0.0;

                //Save the value in results
                results[i].get(j).setQuantumPSOSSE(value);
            }
        }
    }

    public void simulateStandardPSOSSE() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            //StandardPSO nn = new StandardPSO_SSE();
            StandardPSO_SSE nn = new StandardPSO_SSE(3, 22, 30, 2, dataSet);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                nn.iteration();
                double value = nn.getScore(nn.gbest);

                //Save the value in results
                results[i].get(j).setStandardPSOSSE(value);

                double predictionError = getTotalPredictionError(dataSet, nn.gbest.getParticle());
                results[i].get(j).setStandardPSOPE(predictionError);
            }

            if (i == 0) {
                System.out.println("PSO end state");
                for (int k = 0; k < dataSet.size(); k++) {
                    double prediction = nn.gbest.getParticle().compute(dataSet.get(k).getInput()).getData(0);
                    double aim = dataSet.get(k).getIdeal().getData(0);
                    System.out.println(prediction + "," + aim);
                }
            }
        }
    }

    public void simulateStandardPSOPE() {
        //For each data file simulate 
        for (int i = 0; i < datasets.length; i++) {
            MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
            //StandardPSO nn = new StandardPSO_SSE();
            StandardPSO_PE nn = new StandardPSO_PE(3, 22, 30, 2, dataSet);

            //For each iteration calculate the value
            for (int j = 0; j < results[i].epochs; j++) {

                nn.iteration();
                double value = nn.getScore(nn.gbest);

                //Save the value in results
                results[i].get(j).setStandardPSOSSE(value);

                double predictionError = getTotalPredictionError(dataSet, nn.gbest.getParticle());
                results[i].get(j).setStandardPSOPE(predictionError);
            }

            if (i == 4) {
                System.out.println("PSO end state");
                for (int k = 0; k < dataSet.size(); k++) {
                    double prediction = nn.gbest.getParticle().compute(dataSet.get(k).getInput()).getData(0);
                    double aim = dataSet.get(k).getIdeal().getData(0);
                    System.out.println(prediction + "," + aim);
                }
            }
        }
    }
}
