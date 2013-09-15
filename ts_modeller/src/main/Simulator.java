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

import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.util.simple.EncogUtility;
import pso.ChargedPSO;

import pso.FFNNParticle;
import pso.StandardPSO;

import recurrent.Elman;
import recurrent.Jordan;

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
        simulateFFNN();
        simulateElman();
        simulateJordan();
        simulateStandardPSOSSE();
        simulateChargedPSOSSE();

        for (int i = 0; i < results.length; i++) {
            if (stuart) {
                //System.out.println("Output configured to stuart");
                switch (i) {
                    case 0:
                        results[i].printData("Ouput_Brazil.csv");
                        break;
                    case 1:
                        results[i].printData("Ouput_Russia.csv");
                        break;
                    case 2:
                        results[i].printData("Ouput_India.csv");
                        break;
                    case 3:
                        results[i].printData("Ouput_China.csv");
                        break;
                    case 4:
                        results[i].printData("Ouput_SA.csv");
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
                totalPredictionError += error;
            } else {
                double error = aim - prediction;
                totalPredictionError += error;
            }
        }
        return totalPredictionError;
    }

    public void simulateElman() {
        for (int h = 0; h < samples; h++) {
            System.out.println("Executing Elman sample " + h);
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
                    double value = trainMain.getError();
                    results[i].get(j).elmanSSE.add(value);
                    results[i].get(j).elmanPE.add(getTotalPredictionError(dataSet, nn.getParticle()));
                    trainMain.iteration();
                }

                if (i == 0) {
                    System.out.println("Elman end state");
                    for (int k = 0; k < dataSet.size(); k++) {
                        double prediction = nn.getParticle().compute(dataSet.get(k).getInput()).getData(0);
                        double aim = dataSet.get(k).getIdeal().getData(0);
                        System.out.println(prediction);
                    }
                }
            }
        }
    }

    public void simulateJordan() {
        for (int h = 0; h < samples; h++) {
            System.out.println("Executing Jordan sample " + h);
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
                    results[i].get(j).jordanSSE.add(value);
                    results[i].get(j).jordanPE.add(getTotalPredictionError(dataSet, nn.getParticle()));
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
    }

    public void simulateFFNN() {
        for (int h = 0; h < samples; h++) {
            System.out.println("Executing FFNN sample " + h);
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
                    results[i].get(j).FFNNSSE.add(value);
                    results[i].get(j).FFNNPE.add(getTotalPredictionError(dataSet, nn.getParticle()));
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
    }

    public void simulateStandardPSOSSE() {
        for (int h = 0; h < samples; h++) {
            System.out.println("Executing PSO sample " + h);
            //For each data file simulate 
            for (int i = 0; i < datasets.length; i++) {
                MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
                //StandardPSO nn = new StandardPSO();
                StandardPSO nn = new StandardPSO(3, 22, 30, 2, dataSet);

                //For each iteration calculate the value
                for (int j = 0; j < results[i].epochs; j++) {
                    nn.iteration();
                    double value = nn.getScore(nn.gbest);
                    results[i].get(j).standardPSOSSE.add(value);
                    results[i].get(j).standardPSOPE.add(getTotalPredictionError(dataSet, nn.gbest.getParticle()));
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
    }

    public void simulateChargedPSOSSE() {
        for (int h = 0; h < samples; h++) {
            System.out.println("Executing Charged PSO sample " + h);
            //For each data file simulate 
            for (int i = 0; i < datasets.length; i++) {
                MLDataSet dataSet = EncogUtility.loadEGB2Memory(new File(datasets[i]));
                //StandardPSO nn = new StandardPSO();
                ChargedPSO nn = new ChargedPSO(3, 22, 30, 2, dataSet);

                //For each iteration calculate the value
                for (int j = 0; j < results[i].epochs; j++) {
                    nn.iteration();
                    double value = nn.getScore(nn.gbest);
                    results[i].get(j).chargedPSOSSE.add(value);
                    results[i].get(j).chargedPSOPE.add(getTotalPredictionError(dataSet, nn.gbest.getParticle()));
                }

                if (i == 0) {
                    System.out.println("Charged PSO end state");
                    for (int k = 0; k < dataSet.size(); k++) {
                        double prediction = nn.gbest.getParticle().compute(dataSet.get(k).getInput()).getData(0);
                        double aim = dataSet.get(k).getIdeal().getData(0);
                        System.out.println(prediction + "," + aim);
                    }
                }
            }
        }
    }
}
