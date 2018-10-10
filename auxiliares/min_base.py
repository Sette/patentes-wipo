
import csv

PATH = "../../base-wipo/preprocess/"
csv.field_size_limit(10**9)

with open(PATH+'treinamento_min.csv', 'w') as csvfile_out:
   	spamwriter = csv.writer(csvfile_out, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)

   	with open(PATH+'treinamento.csv', 'r') as csvfile_in:
         spamreader = csv.reader(csvfile_in, delimiter=';', quotechar='|')
         i = 0
         for row in spamreader:
            spamwriter.writerow(row)
            i+=1
            print("Carregando %d de 400", i)
            if i == 401:
               print("Fim do treinamento")
               break


with open(PATH+'teste_min.csv', 'w') as csvfile_out:
      spamwriter = csv.writer(csvfile_out, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)

      with open(PATH+'teste.csv', 'r') as csvfile_in:
         spamreader = csv.reader(csvfile_in, delimiter=';', quotechar='|')
         i = 0
         for row in spamreader:
            spamwriter.writerow(row)
            i+=1
            print("Carregando %d de 200", i)
            if i == 201:
               print("Fim do teste")
               break
