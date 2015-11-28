#include <iostream>
#include <complex>
#include <boost/timer/timer.hpp>
#include <gf.h>

using namespace std; 

const int N = 4; 

typedef complex<double> dcomplex; 

int main()
{
   cout << " Starting main " << endl; 

   // Enum that contains names of gf indeces
   enum class MIXED{ w, W };

   // Double-valued 2-dimensional gf
   using mygf_t = gf< double, 2 >;

   // Create object with one fermionic and one bosonic frequency
   mygf_t my_gf( boost::extents[ffreq(N)][bfreq(N)] ); 

   // Initialize values with a lambda function
   my_gf.init( []( const mygf_t::idx_t& idx )->double{ return idx(0) + N * idx(1); } ); 

   // Fill a vector with all possible indeces
   std::vector< mygf_t::idx_t > idx_lst; 
   my_gf.fill_idx_lst( idx_lst ); 

   // Consider the element x,y
   int x = 3;
   int y = 3; 
   mygf_t::idx_t idx( { x, y } ); 

   // Access test
   cout << " Acess Test " << endl; 
   cout << " idx(w) " << idx( MIXED::w ) << endl; 
   cout << " Direct access of gf[x][y] " << my_gf[x][y] << endl; 
   cout << " Acess with idx_t object " << my_gf( idx ) << endl; 
   cout << " Acess with corresponding pos1d " << my_gf( my_gf.get_pos_1d( idx ) ) << endl << endl; 

   // Output idx_t type object and check get_pos_1d and get_idx functions
   cout << " Output Test " << endl; 
   cout << " idx " << idx << endl; 
   cout << " get_idx( gf.pos_1d( idx ) ) " << my_gf.get_idx( my_gf.get_pos_1d( idx ) ) << endl << endl; 

   cout << " Test sum over all container elements " << endl; 
   double val = 0.0; 
   {  // .. with direct access
      boost::timer::auto_cpu_timer t;
      for( int w = -N; w < N; w++ )
	 for( int W = -N; W < N + 1; W++ )
	    val += my_gf[w][W]; 
   }
   cout << " .. using direct acess gf[][] : " << val << endl;  
   val = 0.0; 
   {  // .. with idx_t access
      boost::timer::auto_cpu_timer t;
      for( auto idx : idx_lst )
	 val += my_gf( idx ); 
   }
   cout << " .. using idx acess gf() " << val << endl << endl; 

   // -- Access with idx_t of similar gf_t
   using mygf_other_t = gf< int, 2 >;
   mygf_other_t my_other_gf( boost::extents[ffreq(N+1)][bfreq(N+1)] ); 
   mygf_other_t::idx_t other_idx( { 0, 0} ); 
   mygf_t::idx_t cloned_idx( other_idx ); 
   my_gf ( other_idx ); 

   // --  Abs and Norm
   cout << " my_gf[-1][0] " << my_gf[-1][0] << endl; 
   cout << " abs(my_gf)[-1][0] " << abs(my_gf)[-1][0] << endl; 
   cout << " norm(my_gf) " << norm( my_gf )  << endl; 

   // --  Two gf Operators   // generalize such that g<double> + g<int> possible?
   my_gf += my_gf;     	my_gf + my_gf; 
   my_gf -= my_gf; 	my_gf - my_gf; 
   my_gf *= my_gf;     	my_gf * my_gf; 
   my_gf /= my_gf; 	my_gf / my_gf; 

   // -- Scalar operators
   my_gf += 1.0;     	my_gf + 1.0;	1.0 + my_gf; 
   my_gf -= 1.0; 	my_gf - 1.0;    1.0 - my_gf;
   my_gf *= 1.0;     	my_gf * 1.0;    1.0 * my_gf;
   my_gf /= 1.0; 	my_gf / 1.0;    

   // Example for two-particle vertex
   const int POS_FREQ_COUNT_VERT = 10; 
   const int PATCH_COUNT = 4; 
   const int QN_COUNT = 2; 
   enum class I2P{ w1_in, w2_in, w1_out, k1_in, k2_in, k1_out, s1_in, s2_in, s1_out, s2_out }; 
   class gf_2p_t : public gf< dcomplex, 10 >              ///< Container type for two-particle correlation functions
   {
      public:
	 gf_2p_t():
	    gf< dcomplex, 10 >( boost::extents[ffreq(POS_FREQ_COUNT_VERT)][ffreq(POS_FREQ_COUNT_VERT)][ffreq(POS_FREQ_COUNT_VERT)]
		  [PATCH_COUNT][PATCH_COUNT][PATCH_COUNT]
		  [QN_COUNT][QN_COUNT][QN_COUNT][QN_COUNT] )
      {}
   }; 
   using idx_2p_t = gf_2p_t::idx_t;  

}
