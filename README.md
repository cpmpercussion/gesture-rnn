# Gesture-RNN: A deep model of ensemble interaction in musical gestures

`Gesture-RNN` is a deep model of interaction between an ensemble of improvising computer musicians. Rather than represent notes and rhythms, this approach to musical machine learning focusses on high-level musical gestures.

In particular, `Gesture-RNN` is designed to represent ensemble free-improvised music made on touch-screens. In this kind of ensemble, rather than focussing on melody and harmony, the musicians often work as gestural explorers. Patterns of interaction with the instruments and between the musicians are the most important aspect of the performances. Touch-screen improvisations have been previously categorised in terms of nine simple touch-gestures, and a [large corpus of collaborative touch-screen performances is freely available](https://github.com/anucc/metatone-analysis). This dataset is used as training data in `Gesture-RNN`.

`Gesture-RNN` uses a similar neural network architecture to other creative machine learning systems, such as [folkRNN](https://github.com/IraKorshunova/folk-rnn), Magenta's [musical RNNs](https://github.com/tensorflow/magenta), and  [charRNN](https://github.com/karpathy/char-rnn). It has recently become apparent that recurrent neural networks, which can be equipped with a kind of "memory" to learn long sequences of temporally-related information, can be [unreasonably effective](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). Creative neural network systems are beginning to be a bit of a party trick, like the amusingly bad [NN-generated Christmas carol](https://vimeo.com/192711856). In the case of high-level ensemble interactions, we don't have tools (like music theory) to help us understand and compose them, so a data-driven approach could be much more useful!

Like other systems, `Gesture-RNN` uses long short-term memory (LSTM) cells to preserve information between time-steps and the network is trained to predict the next time-step in a sequence. The difference with this system is that the network is trained to predict how an ensemble would react to a single performer. Training to react appropriately means that the network should learn how ensembles interact, in terms of their high-level gestures.

One application of this system is a kind of "fake" ensemble; one live performer plays on a touch screen, and `Gesture-RNN` is used to generate predictions of the gestures others in an ensemble might use in their reaction. The gestures are then sonified and played back to the live performer. As the live performer is also listening to the generated ensemble, they react to it's performance as well, so the result is a co-creative performance between the human performer and neural ensemble. `Gesture-RNN` has been used in this kind of "neural touch-screen ensemble", there's more details below.

## Learning Ensemble Interactions

Training data for `Gesture-RNN` consists of time-series of gestural classifications from [group improvisations on touch-screen instruments](https://github.com/anucc/metatone-analysis). These time-series consist of a gestural classification for each member of the group at one second intervals. `Gesture-RNN` is designed to predict the ensemble response to a single 'lead' sequence of gestures. So in the case of a gesture recording of a quartet, one player is taken to be the leader, and the network is trained to predict the reaction of the other three players.

In `Gesture-RNN`, the input for the network is the lead player's current gesture, and also the previous gesture of the other ensemble members. The output of the network is the ensemble members predicted reaction to the lead player's gesture. When the network is used to generate possible ensemble reactions, this gestural output is fed back in to the network at the next time-step.

![Gesture-RNN is trained using sequences of gestural classifications from an ensemble. The input for the network is a lead player's current gesture and the ensemble's previous gesture. The output is the predicted response of the ensemble to the lead player.](https://github.com/cpmpercussion/ensemble-performance-deep-models/raw/master/images/nn-ensemble-training.jpg)

In the data-set, there are more than 70 collaborative improvisations, of which 33 are quartets. To train the network, each of player in these quartets is taken as the leader, and the order of the other three players is permuted as the ordering is significant in the input of the network. This results in a large enough set of training data to produce useful outputs. `Gesture-RNN` has also been used to generate duets, where there is one lead player and one follower. For this application, training data can be taken from every pair of performers in the corpus.

## Network Architecture

`Gesture-RNN` uses three layers of 512 [long short-term memory (LSTM) units](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), similarly to other artificial neural networks for learning sequences of creative information such as [folkRNN](https://github.com/IraKorshunova/folk-rnn). These artificial neurons include a kind of "memory" that can be added to, accessed, or erased over multiple evaluations of the neural network. 

`Gesture-RNN` is implemented in [Tensorflow](https://www.tensorflow.org/) and Python. It's tricky to learn how to structure Tensorflow code and the following blog posts and resources were helpful: [WildML: RNNs in Tensorflow, a practical guide](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/), [R2RT: Recurrent Neural Networks in Tensorflow](http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html), [AI Codes: Tensorflow Best Practices](https://github.com/aicodes/tf-bestpractice), [Géron: Hands-On Machine Learning with Scikit-Learn and Tensorflow](http://shop.oreilly.com/product/0636920052289.do)

## Usage

To run `Gesture-RNN` you'll need `numpy`, `pandas`, `tensorflow`, and `h5py`.

To train the network run:

	python gesture-RNN.py --train

This will generate the training data and train the network for 30 epochs. On a Nvidia GTX 1080 GPU this takes about 9 hours.

To generate some sample data run:

	python gesture-RNN.py --generate

`Gesture-RNN` is also used as a module in [Metatone Classifier](https://github.com/cpmpercussion/MetatoneClassifier/tree/soundobject-player) as part of the neural touch-screen ensemble.

## Example Output

Here's some example output from `Gesture-RNN`. In these plots, a real lead performance (in red) was used as the input and the ensemble performers (other colours) were generated by `Gesture-RNN`. The upper plot uses the duet model and the lower plot is the quartet model. Each level on the y-axis in these plots represents a different musical gesture performed on the touch-screens.

![Duet Example](https://github.com/cpmpercussion/ensemble-performance-deep-models/raw/master/images/duet-example.png)

![Quartet Example](https://github.com/cpmpercussion/ensemble-performance-deep-models/raw/master/images/quartet-example.png)

## Neural Touch-Screen Ensemble

A fun application of `Gesture-RNN` is to make a kind of "fake ensemble". The lead-player part is provided by a human performer playing a touch-screen instrument. `Gesture-RNN` then generates the ensemble parts which are sonified to provide a predicted ensemble response. This system gives some of the experience of performing in a touch-screen improvisation ensemble, and has the added bonus of allowing direct evaluation of the performance of the RNN through live interaction.

I've used `Gesture-RNN` along with [Metatone Classifier](https://github.com/cpmpercussion/MetatoneClassifier), and the [PhaseRings](https://github.com/cpmpercussion/PhaseRings) iPad music app to create a Neural Touch-Screen Ensemble. The performer plays on one iPad and the output from `Gesture-RNN` is sonified (and visualised) through two other creating an embodied co-creative experience.

![The Neural Touch-Screen Ensemble in action! [Video here!](https://youtu.be/6eg5VSRqIDA)](https://github.com/cpmpercussion/ensemble-performance-deep-models/raw/master/images/neural-touch-screen-band-small.jpg)

The system works as follows: A live performer plays on a touch-screen instrument and Metatone Classifier generates their current gesture. `Gesture-RNN` then generates response gestures from the neural ensemble. A touch-synthesiser system searches the data-corpus for touch-sequences matching each particular gesture and sends them to other touch-screen devices. Finally, these other devices play back the sounds of these touch-sequences as they are streamed to them.

![Live Performance System](https://github.com/cpmpercussion/ensemble-performance-deep-models/raw/master/images/live-performance-system.jpg)

This system is [currently working](https://youtu.be/6eg5VSRqIDA) well enough for in-lab demonstrations and we're hoping to show it off at a few events soon!

<!-- <iframe width="560" height="315" src="https://www.youtube.com/embed/6eg5VSRqIDA" frameborder="0" allowfullscreen></iframe>
 -->

## Todo!

Lots to do with `Gesture-RNN`. This system is still under-development as part of efforts to use RNNs in new creative situations. Any suggestions about the network structure and ways to evaluate the results are welcome!

Here's some things I'm working on:

- Automate duet model training.
- Use best-practices for storing and generating training data.
- Develop trio model.
- Evaluate trained models in performances and installations.
