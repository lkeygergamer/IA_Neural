Teste no Google Colab.

Roxas 1 Berçario 
```python
# Direitos Autorais © 2024 Adilson Oliveira. Todos os direitos reservados.
# Este código é propriedade intelectual de Adilson Oliveira, sendo proibido
# qualquer uso, modificação ou redistribuição sem autorização explícita do autor.
# Este código foi desenvolvido sem dependência de bibliotecas de terceiros.
```

```python
# Conectar ao Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Criar pasta para salvar os arquivos
import os
save_path = '/content/drive/MyDrive/IA_NeuralNetwork'
os.makedirs(save_path, exist_ok=True)

# Propriedade Intelectual de Adilson Oliveira.
# Funções matemáticas básicas
def matmul(A, B):
    result = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def exp(x):
    n_terms = 10  # Número de termos na série de Taylor
    result = 1.0
    term = 1.0
    for i in range(1, n_terms):
        term *= x / i
        result += term
    return result

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Funções de ativação
class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def sigmoid_derivative(output):
        return output * (1 - output)

# Camadas da rede neural
class NeuralLayer:
    def __init__(self, n_input, n_neurons, seed=0):
        # Inicializa os pesos de maneira determinística
        self.weights = [[self.deterministic_value(i, j) for j in range(n_neurons)] for i in range(n_input)]
        self.biases = [self.deterministic_value(i) for i in range(n_neurons)]
    
    def deterministic_value(self, i, j=None):
        # Função determinística para inicializar pesos e vieses (ex: fórmula simples de multiplicação)
        result = 0.1  # Valor base
        if j is not None:
            result *= (i + 1) * (j + 1) * 0.5  # Para pesos, consideramos dois parâmetros (i, j)
        else:
            result *= (i + 1) * 0.5  # Para vieses, apenas o índice do neurônio
        return result
    
    def forward(self, inputs):
        self.inputs = inputs
        weighted_sum = matmul(inputs, self.weights)
        self.output = [[ActivationFunctions.sigmoid(weighted_sum[i][j] + self.biases[j]) for j in range(len(self.biases))] for i in range(len(weighted_sum))]
        return self.output

    def backward(self, d_output, learning_rate):
        d_activation = [[ActivationFunctions.sigmoid_derivative(self.output[i][j]) * d_output[i][j] for j in range(len(d_output[0]))] for i in range(len(d_output))]
        d_weights = matmul(transpose(self.inputs), d_activation)
        
        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] -= learning_rate * d_weights[i][j]
        
        for i in range(len(self.biases)):
            self.biases[i] -= learning_rate * sum([d_activation[j][i] for j in range(len(d_activation))])
        
        return matmul(d_activation, transpose(self.weights))

# Rede Neural Principal
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d_loss, learning_rate):
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = sum([(y[i][0] - output[i][0]) ** 2 for i in range(len(y))]) / len(y)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

            d_loss = [[2 * (output[i][0] - y[i][0]) / len(y) for _ in range(len(output[0]))] for i in range(len(output))]
            self.backward(d_loss, learning_rate)

    def save_model(self, path):
        with open(path, 'w') as f:
            for layer in self.layers:
                f.write(f"Weights: {layer.weights}\n")
                f.write(f"Biases: {layer.biases}\n")

# Dados de exemplo (XOR)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# Definir a estrutura da rede
layers = [
    NeuralLayer(2, 3),
    NeuralLayer(3, 1)
]

# Inicializar e treinar a rede
nn = NeuralNetwork(layers)
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Salvar o modelo no Google Drive
model_path = os.path.join(save_path, 'modelo_rede_neural.txt')
nn.save_model(model_path)
print(f"Modelo salvo em: {model_path}")

# Interface de conversa com a IA
def conversar_com_ia():
    print("Digite dois valores binários separados por espaço (por exemplo, '1 0') ou 'sair' para encerrar:")
    while True:
        entrada = input("Entrada: ")
        if entrada.lower() in ('sair', 'exit'):
            break
        try:
            valores = [[int(x) for x in entrada.split()]]
            resposta = nn.forward(valores)
            print(f"Resposta da IA: {resposta[0][0]}")
        except:
            print("Entrada inválida. Tente novamente.")

# Chamar a função de conversa
conversar_com_ia()
```

Este projeto é a personificação da inovação genuína, desenvolvido com precisão e autenticidade, cada linha de código sendo fruto exclusivo do meu próprio intelecto. Nenhum detalhe foi deixado ao acaso, cada componente reflete a essência da minha visão, comprometida com a excelência e a integridade da propriedade intelectual.

Assinatura: Adilson Oliveira 

