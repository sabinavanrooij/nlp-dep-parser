class Token:
    def __init__(self, index, word, POSTag, head, label):
        self.index = index
        self.word = word
        self.POSTag = POSTag
        self.head = head
        self.label = label
    
    def __str__(self):        
        return "Index: {0}, Word: {1}, POSTag: {2}, Head: {3}, Label: {4}".format(self.index, self.word, self.POSTag, self.head, self.label)

class TrainingExample:
    def __init__(self):
        self.tokens = {}
    
    def addToken(self, token):
        self.tokens[token.index] = token
    
    def getToken(self, index):
        return self.tokens[index]
    
    def __str__(self):
        strList = []
        for k in self.tokens:
            strList.append(str(self.tokens[k]))
        return '\n'.join(strList)
        
class ConlluFileReader:
    def __init__(self, filePath):
        self.filePath = filePath
        
    def readTrainingSet(self):
        f = open(self.filePath, 'r')
        trainingSet = []
        trainingExample = TrainingExample()
        
        for line in f.readlines():            
            if line.startswith('#'):
                continue
            
            if line.isspace(): # end of the sentence
                trainingSet.append(trainingExample)
                trainingExample = TrainingExample()
                continue
                
            items = line.split('\t')
            index = float(items[0]) # this can be a float if the word is implicit in the sentence
            head = items[6] # this can be '_' for implicit words that were added, change to -1
            if head == '_':
                head = -1
            else:
                head = float(head)
            
            trainingExample.addToken(Token(index, items[1], items[3], head, items[7]))
        
        f.close()
        
        return trainingSet
