/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package recurrent;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.pattern.ElmanPattern;

/**
 *
 * @author stuart
 */
public class Elman {

    BasicNetwork nn;
    int numInputs, numHidden, numOutputs;

    public Elman(int in, int hidden, int out) {
        numInputs = in;
        numHidden = hidden;
        numOutputs = out;
        nn = constructParticle();
    }

    private BasicNetwork constructParticle() {
        ElmanPattern pattern = new ElmanPattern();
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
