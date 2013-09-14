/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package recurrent;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.pattern.ElmanPattern;
import org.encog.neural.pattern.JordanPattern;

/**
 *
 * @author stuart
 */
public class Jordan {

    BasicNetwork nn;
    int numInputs, numHidden, numOutputs;

    public Jordan(int in, int hidden, int out) {
        numInputs = in;
        numHidden = hidden;
        numOutputs = out;
        nn = constructParticle();
    }

    private BasicNetwork constructParticle() {
        // construct an Jordan type network
        JordanPattern pattern = new JordanPattern();
        pattern.setActivationFunction(new ActivationSigmoid());
        pattern.setInputNeurons(numInputs);
        pattern.addHiddenLayer(numHidden);
        pattern.setOutputNeurons(numOutputs);
        return (BasicNetwork) pattern.generate();
    }

    public BasicNetwork getParticle() {
        return nn;
    }
}
