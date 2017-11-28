class Token:
    def __init__(self, index, word, POSTag, head, label):
        self.index = index
        self.word = word
        self.POSTag = POSTag
        self.head = head
        self.label = label
    
    def __str__(self):        
        return "Index: {0}, Word: {1}, POSTag: {2}, Head: {3}, Label: {4}".format(self.index, self.word, self.POSTag, self.head, self.label)

class SentenceDependencies:
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
    

commentSymbol = '#'
itemsSeparator = '\t'
undefinedField = '_'
    
class ConlluFileReader:
    def __init__(self, filePath):
        self.filePath = filePath
        
    def readTrainingSet(self):
        f = open(self.filePath, 'r')
        trainingSet = []
        trainingExample = SentenceDependencies()
        
        for line in f.readlines():            
            if line.startswith(commentSymbol):
                continue
            
            if line.isspace(): # end of the sentence
                trainingSet.append(trainingExample)
                trainingExample = SentenceDependencies()
                continue
                
            items = line.split(itemsSeparator)
            
            # this can be a float or a range if the word is implicit in the sentence
            if not items[0].is_integer():
                continue
                
            index = int(items[0]) 
            head = items[6] # this can be '_' for implicit words that were added, change to -1
            if head == undefinedField:
                head = -1
            else:
                head = float(head)
            
            trainingExample.addToken(Token(index, items[1], items[3], head, items[7]))
        
        f.close()
        
        return trainingSet

class ConlluFileWriter:
    def __init__(self, filePath):
        self.filePath = filePath
    
    def getFormattedIndex(self, index):
        if index.is_integer():
            return str(int(index))
        return str(index)
        
    def write(self, sentenceDependencies):
        f = open(self.filePath, 'w')
        lines = []        
        for sentenceDep in sentenceDependencies:
            sentence = []            
            itemsLines = []
            for k, v in sentenceDep.tokens.items():
                sentence.append(v.word)
                items = []
                
                items.append(self.getFormattedIndex(v.index))
                items.append(v.word)
                items.append(undefinedField)
                items.append(v.POSTag)
                items.append(undefinedField)
                items.append(undefinedField)
                if v.head == -1:
                    items.append(undefinedField)
                else:
                    items.append(self.getFormattedIndex(v.head))
                items.append(v.label)
                items.append(undefinedField)
                items.append(undefinedField)
                itemsLines.append(itemsSeparator.join(items))            
            
            lines.append("{0} text = {1}".format(commentSymbol, ' '.join(sentence)))
            lines.append("{0}\n".format('\n'.join(itemsLines)))
            
        f.write('\n'.join(lines))
        f.close()
