#include "trabalho.h"

User *generateUser(char *buffer, int nRatings)
{
  int pos = 0, ratePos = 0, state = 0, size;
  float rateAux = -1.0f;
  User *newUser = (User*)malloc(sizeof(User));
  
  newUser->name = "None";
  newUser->rating = (float*) calloc(nRatings, sizeof(float));
  
  char *tempBuffer = "";

  size = strlen(buffer);

  do {

    if(state == 0 && buffer[pos] == ';')
    {
      tempBuffer = buffer;
      strtok(tempBuffer, ";");
      newUser->name = (char*)malloc(strlen(tempBuffer) * sizeof(char));
      newUser->name = strcpy(newUser->name, tempBuffer);
      state = 1;
    }
    else if(state == 1 && buffer[pos] != ';')
    {
		// ratings
        switch(buffer[pos])
        {
			case '?': 
				newUser->rating[ratePos] = -1;
				break;
				
			default:
				sscanf(&buffer[pos], "%f", &rateAux);
				newUser->rating[ratePos] = rateAux;
				rateAux = -1.0f;
				break;
		}
		ratePos++;
    }
    
    pos++;
    
  }while(pos < size);
   
  return newUser;

}

int findAmmountOfRatesByUser(User *target, int nElements)
{
  int total = 0;

  for(int i=0; i<nElements; i++)
  {
    if(target->rating[i] != '?' && target->rating[i] != -1)
    {
      total++;
    }
  }

  return total;
}

// cálculo em GPU da média dos rates feitos pelo usuário
__global__ void cudaCalcUserAverageRate(float *ratings, int *dev_nElements, float result)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < *dev_nElements)
	{
		result = 0.0f;
		int count = 0;
		
		for(int j=0; j < *dev_nElements; j++)
		{
			if(ratings[j] != '?' && ratings[j] != -1)
			{
				count++;
				result += ratings[j];
			}
		}
		
		result /= count;
	}
}


float calcUserAverageRate(User *target, int nElements)
{
  float average = 0.0f;
  int count = 0;

  for(int i=0; i<nElements; i++)
  {
    if(target->rating[i] != '?' && target->rating[i] != -1)
    {
      count++;
      average+= target->rating[i];
    }
  }

  average /= count;

  return average;
}

__global__ void cudaCalPearsonCorrelation(float *ratingsA, float *ratingsB, int *dev_nElements, float pearson)
{
	

	int i = threadIdx.x;
	

	if(i < *dev_nElements)
	{	
		int nRates = 0;
		float pearsonNum = 0.0f, pearsonDen = 0.0f; // pearson = 0.0f
		float tempA = 0.0f, tempB = 0.0f, tempC = 0.0f, tempD = 0.0f;
	
		if(ratingsA[i] == -1 || ratingsB[i] == -1)
		{
		  // DO NOTHING, someone hasn't evaluated an item
		}
		else // if both users have rated this item
		{
		  nRates++;
		  tempA += ratingsA[i]; //stores the Sum of the rates from A
		  tempB += ratingsB[i]; //stores the Sum of the rates from B
		  pearsonNum += ratingsA[i] * ratingsB[i];
		  tempC += powf(ratingsA[i], 2); // stores the Sum of the rate² from A
		  tempD += powf(ratingsB[i], 2); // stores the Sum of the rate² from B
		}

		pearsonNum -= (tempA * tempB) / nRates; // sumXY - ((sumX)(sumY)/n)
		tempC -= powf(tempA, 2) / nRates; // sumX² - ((sumX)²/n)
		tempD -= powf(tempB, 2) / nRates; // sumY² - ((sumY)²/n)
		pearsonDen = sqrtf(tempC * tempD);

		pearson = pearsonNum / pearsonDen;
	}
}

float calcPearsonCorrelation(User *a, User *b, int nElements)
{

  float pearson = 0.0f, pearsonNum = 0.0f, pearsonDen = 0.0f;
  float tempA = 0.0f, tempB = 0.0f, tempC = 0.0f, tempD = 0.0f;

  // int ratedItens[MAXITENS]; // Store the itens that are rated by both users | 1 = rated by both, 0 otherwise
  int nRates = 0;


  // loop that verify the itens that has been rated by each user and set the itens that can be used to calculate the similarity
  for(int i=0; i<nElements; i++)
  {
    if(a->rating[i] == -1 || b->rating[i] == -1)
    {
      continue; // someone hasn't evaluated an item
    }
    else // if both users have rated this item
    {
      // ratedItens[i] = 1;
      nRates++;
      tempA += a->rating[i]; //stores the Sum of the rates from A
      tempB += b->rating[i]; //stores the Sum of the rates from B
      pearsonNum += a->rating[i] * b->rating[i];
      tempC += pow(a->rating[i], 2); // stores the Sum of the rate² from A
      tempD += pow(b->rating[i], 2); // stores the Sum of the rate² from B
    }
  }

  pearsonNum -= (tempA * tempB) / nRates; // sumXY - ((sumX)(sumY)/n)
  tempC -= pow(tempA, 2) / nRates; // sumX² - ((sumX)²/n)
  tempD -= pow(tempB, 2) / nRates; // sumY² - ((sumY)²/n)
  pearsonDen = sqrt(tempC * tempD);

  pearson = pearsonNum / pearsonDen;
  //printf("Pearson entre %s e %s = %.2f\n", a->name, b-> name, pearson);
	
  return pearson;
}

float predictRateByUser(User **array, int a_size, User *desired, int itemID, int nElements)
{
	float pred = -0.5f, demRes = 0, numRes = 0, pearson = 0, des_average = 0, cur_average = 0, temp_average = 0;
	float *dev_ratings; 
	
	cudaMalloc(&dev_ratings, nElements*sizeof(float));
	cudaMemcpy(dev_ratings, desired->rating, nElements*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaCalcUserAverageRate<<<32, 1>>>(desired->rating, &nElements, des_average);
	
	cudaFree(dev_ratings);
	
	for(int i=0; i<a_size; i++)
	{
		if(desired == array[i])
		{
			continue; // represents itself
		}
		else if(array[i]->rating[itemID] == -1)
		{			
			continue; // marks to skip in case it hasn't evaluate
		}
		else
		{
			cudaCalPearsonCorrelation<<<32, 1>>>(desired->rating, array[i]->rating, &nElements, pearson);
			if(pearson >= 0.7) 
			{
				demRes += pearson;
				cudaMalloc(&dev_ratings, nElements*sizeof(float));
				cudaMemcpy(dev_ratings, desired->rating, nElements*sizeof(float), cudaMemcpyHostToDevice);
	
				cudaCalcUserAverageRate<<<32, 1>>>(desired->rating, &nElements, temp_average);

				cur_average = array[i]->rating[itemID] - temp_average;

				cudaFree(dev_ratings);

				numRes += (pearson * cur_average);
			}
		}

	}
	
	pred = des_average + (numRes / demRes);
	
  return pred;
}


int main(int argc, char *argv[])
{

  int targetItemId = -1, nElements = 0, timesReaded = 0, maxLines = 0;
  char *targetUserName, readBuffer[1024], charBuffer, *filename;
  FILE *fp;
  User *targetUser = NULL;
  
  if(argc != 4)
  {
    fprintf(stderr,"Formato : %s Arquivo (string) Usuário (string) Item# (int)\n",argv[0]);
    return 1;
  }
  
  filename = argv[1];
  targetUserName = argv[2];
  sscanf(argv[3], "%d", &targetItemId);
  targetItemId--;
  
  fp = fopen(filename, "r");
  if(fp == NULL)
  {
    printf("File couldn't be loaded!\n");
    return 0;
  }
  
  while(feof(fp) == 0)
  {
    charBuffer = fgetc(fp);
    if(charBuffer == '\n')
    {
      maxLines++; // ammount of rows on the file
    }
  }

  rewind(fp); // reset buffer position
  maxLines--; // removes the header line

  fscanf(fp, "%s", readBuffer); // header line
  int size = strlen(readBuffer);

  for(int i=0; i<size; i++)
  {
    if(readBuffer[i] == ';')
    {
      nElements++; // stores the ammount of columns in the file
    }
  }
  
  if(targetItemId > nElements)
  {
	  fprintf(stderr,"Escopo incorreto!\nNão existe a coluna %d, este arquivo possui apenas %d colunas!\n\n", targetItemId, nElements );
	  return EXIT_FAILURE;
  }

  User *usersArray[maxLines];
  
  while(feof(fp) == 0 && timesReaded < maxLines)
  {
    fscanf(fp, "%s", readBuffer);

    if(readBuffer[0] != '\0')
    {
	  usersArray[timesReaded] = generateUser(readBuffer, nElements);
      timesReaded++;
      readBuffer[0] = '\0';
    }
  }
  
  fclose(fp);
  timesReaded = 0; // reseta o valor
  
  // finding the user
  for(int i=0; i<maxLines; i++)
  {
	  if(strcmp(usersArray[i]->name, targetUserName) == 0)
	  {
		  targetUser = usersArray[i];
	  }
	  // contabiliza o # de vezes que esse valor foi medido
	  if(usersArray[i]->rating[targetItemId] != -1)
	  {
		  timesReaded++;
	  }
  }
    
  if(targetUser == NULL)
  {
	  printf("Usuário não encontrado!\n");
	  return EXIT_FAILURE;
  }
  else
  {
	  printf("----------------------| Usuário %s |----------------------\n", targetUser->name);
	  printf("Avaliou %d Itens.\n", findAmmountOfRatesByUser(targetUser, nElements));
	  // usuário avaliou o item
	  if(targetUser->rating[targetItemId] != -1)
	  {
		  printf("Avaliou o Item solicitado: %.2f.\n", targetUser->rating[targetItemId]);
	  }
	  // usuário não avaliou o item
	  else
	  {
		  float pred1 = -1.0f, pred2 = -1.0f;
		  pred1 = predictRateByUser(usersArray, maxLines, targetUser, targetItemId, nElements);
		  printf("Não avaliou o Item Solicitado!!!\n\t|-Previsão por Usuário: %.2f\n\t|-Previsão por Item: %.2f\n", pred1, pred2);
	  }
	  printf("O Item Pesquisado foi avaliado %d vezes\n", timesReaded);
	  printf("--------------------------------------------------------\n");
	  
	  return EXIT_SUCCESS;
  }
   
}
