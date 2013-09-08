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
    int numInputs, numHidden, numOutputs;

    FFNNParticle(int in, int hidden, int out) {
        numInputs = in;
        numHidden = hidden;
        numOutputs = out;
        particle = constructParticle();
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
}
