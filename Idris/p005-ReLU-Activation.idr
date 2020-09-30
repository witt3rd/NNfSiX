{-
 - Associated YT tutorial: https://youtu.be/TEWy9vZcxW4
 -
 - Launch REPL with:
 - > idris -p contrib -p effects
 -
-}
import Util

-- import numpy as np
-- import nnfs
-- from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c
--
-- nnfs.init()
--
-- X, y = spiral_data(100, 3)
--
--
-- class Layer_Dense:
--     def __init__(self, n_inputs, n_neurons):
--         self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
--         self.biases = np.zeros((1, n_neurons))
--     def forward(self, inputs):
--         self.output = np.dot(inputs, self.weights) + self.biases
--
--
-- class Activation_ReLU:
--     def forward(self, inputs):
--         self.output = np.maximum(0, inputs)
--
--
-- layer1 = Layer_Dense(2,5)
-- activation1 = Activation_ReLU()
--
-- layer1.forward(X)
--
-- #print(layer1.output)
-- activation1.forward(layer1.output)
-- print(activation1.output)

||| Inputs (samplex x inputs)
X : Matrix 3 4 Double
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

||| Data structure for a dense layer
record Layer_Dense (n_inputs : Nat) (n_neurons : Nat) where
  constructor MkLayer_Dense
  weights : Matrix n_inputs n_neurons Double
  biases : Vect n_neurons Double
-- Note: can't make output a field, since we don't don't know
--       how many samples we'll be given

||| Smart constructor for a dense layer
newLayer_Dense : (n_inputs : Nat) ->
                 (n_neurons : Nat) ->
                 Eff (Layer_Dense n_inputs n_neurons) [RND]
newLayer_Dense n_inputs n_neurons = pure $
  MkLayer_Dense
    !(rndMat n_inputs n_neurons (-1) 1)
    (zeroVect n_neurons)

||| Feed inputs into the layer and compute outputs
forward : Layer_Dense i n -> Matrix m i Double -> Matrix m n Double
forward layer inputs = (inputs <> (weights layer)) <+ (biases layer)

main : IO ()
main = do
  let layer1 = runPure $ newLayer_Dense 4 5
  let layer2 = runPure $ newLayer_Dense 5 2
  let output1 = forward layer1 X
  -- printLn $ output1
  let output2 = forward layer2 output1
  printLn $ output2
