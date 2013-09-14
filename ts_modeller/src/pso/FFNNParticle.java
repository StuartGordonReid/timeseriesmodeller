/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pso;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.pattern.FeedForwardPattern;

/**
 *
 * @author stuart
 */
public class FFNNParticle {

    BasicNetwork particle;
    double velocity;
    double localBestScore;
    double[] localBest;
    int numInputs, numHidden, numOutputs;

    public FFNNParticle(int in, int hidden, int out) {
        numInputs = in;
        numHidden = hidden;
        numOutputs = out;
        particle = constructParticle();
        localBestScore = Double.MAX_VALUE;
    }

    private BasicNetwork constructParticle() {
        FeedForwardPattern pattern = new FeedForwardPattern();
        pattern.setActivationFunction(new ActivationSigmoid());
        pattern.setInputNeurons(numInputs);
        pattern.addHiddenLayer(numHidden);
        pattern.setOutputNeurons(numOutputs);
        return (BasicNetwork) pattern.generate();
    }

    public BasicNetwork getParticle() {
        return particle;
    }

    public double[] getLocalBestPosition() {
        return localBest;
    }

    public void setLocalBestPosition(double[] localBestPosition) {
        this.localBest = localBestPosition;
    }

    public double getLocalBestScore() {
        return localBestScore;
    }

    public void setLocalBestScore(double localBestScore) {
        this.localBestScore = localBestScore;
    }

}
