#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <math.h>
using namespace std;

int main(int argc, char** argv)
 {

        if (argc!= 6){
                cout<<"Not enough parameters";
                return 0;
        }

	int i=0;
	int j=0;

        uint32_t m= atoi(argv[1]);
        cout<<"m is "<<m<<endl<<endl;

        uint32_t n = atoi(argv[2]);
        cout<<"n is "<< n << endl<<endl;

	uint32_t image_depth= atoi(argv[3]);
	cout<<"Image depth is "<< image_depth << endl<<endl;

        uint32_t k =atoi(argv[4]);
        cout<<"K is"<<k<<endl<<endl;

	uint32_t kernel_depth= atoi(argv[5]);
        cout<<"kernel depth is "<< kernel_depth << endl<<endl;

	uint32_t k_center= (k*k)/2;
	//int c,d,mm,nn,ii,jj;
        int kernelcenterX= k/2;
        int kernelcenterY= k/2;
	int max=20;
	int p_m= ceil(m/2);
	int p_n= ceil(n/2);

	float** R_image= new float*[m];
	float** G_image= new float*[m];
	float** B_image= new float*[m];


        float** R_kernel= new float*[k];
        float** G_kernel= new float*[k];
	float** B_kernel= new float*[k];

	float** output= new float*[m];
	float** pooling_output= new float*[(p_m)];


	for( i=0;i<m;i++)
	{
		R_image[i]= new float[n];
		G_image[i]= new float[n];
		B_image[i]= new float[n];
		output[i]= new float[n];
		pooling_output[i]= new float[(p_n)];
	}


	for( i=0;i<k;i++)
        {
                R_kernel[i]= new float[k];
		G_kernel[i]= new float[k];		
		B_kernel[i]= new float[k];
        }


	for(i=0; i<m; i++)
	{
		for( j=0;j<n;j++)
		{
			R_image[i][j]= 1;
			G_image[i][j]= 1;
			B_image[i][j]= 1;
			output[i][j]= 0;
		}
	}


	 for(i=0; i<k; i++)
        {
                for(j=0;j<k;j++)
                {
                        R_kernel[i][j]= 1;
                        G_kernel[i][j]= 1;
                        B_kernel[i][j]= 1;
                }
        }



/*	 for(i=0; i<m; i++)
        {
                for(j=0;j<n;j++)
                {
                    cout<<" "<<R_image[i][j];
                      
                }
		cout<<endl;
        }

*/	

	/* Start Convolution*/
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(6); // Use 4 threads for all consecutive parallel region


	auto start_time = chrono::high_resolution_clock::now();
	#pragma omp parallel for schedule(dynamic,1000) collapse(2)	
	for(int rows=0;rows<m;rows++)
	{
		for(int columns=0;columns<n;columns++)
		{
			for(int krows=0;krows<k;krows++)
			{
				int mm = k - 1 - krows;

				for(int kcolumns=0; kcolumns<k;kcolumns++)
				{
					int nn = k - 1 - kcolumns; 					
					int ii = rows + (krows - kernelcenterY);
                                        int jj = columns + (kcolumns - kernelcenterX);
	
					// cout<<"ii and jj values are"<<ii<<jj<<endl;
					if(ii>=0 && ii<m && jj>=0 && jj<n)
					{
					
						// cout<<"ii and jj values are"<<ii<<jj<<endl;
						 output[rows][columns] += R_image[ii][jj] * R_kernel[mm][nn];		 
						 output[rows][columns] += G_image[ii][jj] * G_kernel[mm][nn];
						 output[rows][columns] += B_image[ii][jj] * B_kernel[mm][nn];
					
					}
				}
			}
		}
	}

	auto end_time = chrono::high_resolution_clock::now();

	cout <<"The time in microseconds "<< chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()<<endl ;

 	//cout<<"The final matrix after rectifiled linear unit is";
	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			if(output[i][j]<0)
			{
				output[i][j]=0;
			}
			if(output[i][j]>max)
			{
				output[i][j]=max;
			}
		}
	}

	 cout<<"The final matrix after rectifiled linear unit is";

	 for(i=0;i<m;i++)
        {
                for(j=0;j<n;j++)
                {
			cout<<" "<<output[i][j];
		}
		cout<<endl;
	}

	int max1, max2;

	for(i=0; i<m; i+=2)
	{
		for(j=0;j<n;j+=2)
		{	
			if (i+1<m && j+1<n)
			{			
				max1 = output[i][j]>=output[i][j+1]? output[i][j]:output[i][j+1];
				max2 = output[i+1][j]>=output[i+1][j+1]? output[i+1][j]:output[i+1][j+1];
				max1 = max>=max1? max:max1;				
			}
			else if (i+1==m && j+1==n)
			{
				max1 = output[i][j];
			}
			else if (i+1==m)
			{
				max1 = output[i][j]>=output[i][j+1]?output[i][j]:output[i][j+1];
			}
			else if (j+1==n)
			{
				max1 = output[i][j]>=output[i+1][j]?output[i][j]:output[i+1][j];
			}
			pooling_output[i/2][j/2] = max1;

			
		}
		
	}  				

	 cout<<"The final matrix after Pooling layer is";

	for(i=0;i<p_m;i++)
        {
                for(j=0;j<p_n;j++)
                {
                        cout<<" "<<pooling_output[i][j];
                }
                cout<<endl;
        }

	





}


