/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package main;

import java.io.File;
import java.util.ArrayList;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.Greedy;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.ml.train.strategy.StopTrainingStrategy;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.util.simple.EncogUtility;
import pso.FFNNParticle;

/**
 *
 * @author stuart
 */
public class Simulator {

    String[] countries;

    Simulator() {
        countries = new String [] {
        "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/BRAZIL_train.egb",
        "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/RUSSIA_train.egb",
        "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/INDIA_train.egb",
        "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/CHINA_train.egb",
        "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/SOUTHAFRICA_train.egb"};
    }

    public void start() {
        for (int i = 0; i < 1; i++) {
            
            MLDataSet country = EncogUtility.loadEGB2Memory(new File(countries[i]));

            FFNNParticle nn = new FFNNParticle(22, 30, 1);
            CalculateScore score = new TrainingSetScore(country);

            MLTrain trainAlt = new NeuralSimulatedAnnealing(nn.getParticle(), score, 10, 2, 100);
            MLTrain trainMain = new Backpropagation(nn.getParticle(), country, 0.001, 0.0);
            StopTrainingStrategy stop = new StopTrainingStrategy();

            trainMain.addStrategy(new Greedy());
            trainMain.addStrategy(new HybridStrategy(trainAlt));
            trainMain.addStrategy(stop);

            int epoch = 0;
            while (!stop.shouldStop()) {
                trainMain.iteration();
                System.out.println(trainMain.getError());
                epoch++;
            }
        }
    }
}
