/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__can
#define _nrn_initial _nrn_initial__can
#define nrn_cur _nrn_cur__can
#define _nrn_current _nrn_current__can
#define nrn_jacob _nrn_jacob__can
#define nrn_state _nrn_state__can
#define _net_receive _net_receive__can 
#define rates rates__can 
#define states states__can 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gcanbar _p[0]
#define gcanbar_columnindex 0
#define m _p[1]
#define m_columnindex 1
#define h _p[2]
#define h_columnindex 2
#define cai _p[3]
#define cai_columnindex 3
#define cao _p[4]
#define cao_columnindex 4
#define Dm _p[5]
#define Dm_columnindex 5
#define Dh _p[6]
#define Dh_columnindex 6
#define ica _p[7]
#define ica_columnindex 7
#define gcan _p[8]
#define gcan_columnindex 8
#define _g _p[9]
#define _g_columnindex 9
#define _ion_cai	*_ppvar[0]._pval
#define _ion_cao	*_ppvar[1]._pval
#define _ion_ica	*_ppvar[2]._pval
#define _ion_dicadv	*_ppvar[3]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_KTF(void);
 static void _hoc_alpm(void);
 static void _hoc_alph(void);
 static void _hoc_betm(void);
 static void _hoc_beth(void);
 static void _hoc_efun(void);
 static void _hoc_ghk(void);
 static void _hoc_h2(void);
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_can", _hoc_setdata,
 "KTF_can", _hoc_KTF,
 "alpm_can", _hoc_alpm,
 "alph_can", _hoc_alph,
 "betm_can", _hoc_betm,
 "beth_can", _hoc_beth,
 "efun_can", _hoc_efun,
 "ghk_can", _hoc_ghk,
 "h2_can", _hoc_h2,
 "rates_can", _hoc_rates,
 0, 0
};
#define KTF KTF_can
#define _f_betm _f_betm_can
#define _f_alpm _f_alpm_can
#define _f_beth _f_beth_can
#define _f_alph _f_alph_can
#define alpm alpm_can
#define alph alph_can
#define betm betm_can
#define beth beth_can
#define efun efun_can
#define ghk ghk_can
#define h2 h2_can
 extern double KTF( double );
 extern double _f_betm( double );
 extern double _f_alpm( double );
 extern double _f_beth( double );
 extern double _f_alph( double );
 extern double alpm( double );
 extern double alph( double );
 extern double betm( double );
 extern double beth( double );
 extern double efun( double );
 extern double ghk( double , double , double );
 extern double h2( double );
 /* declare global and static user variables */
#define hinf hinf_can
 double hinf = 0;
#define ki ki_can
 double ki = 0.001;
#define minf minf_can
 double minf = 0;
#define tauh tauh_can
 double tauh = 0;
#define taum taum_can
 double taum = 0;
#define usetable usetable_can
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_can", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "ki_can", "mM",
 "gcanbar_can", "mho/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "ki_can", &ki_can,
 "minf_can", &minf_can,
 "hinf_can", &hinf_can,
 "taum_can", &taum_can,
 "tauh_can", &tauh_can,
 "usetable_can", &usetable_can,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"can",
 "gcanbar_can",
 0,
 0,
 "m_can",
 "h_can",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 10, _prop);
 	/*initialize range parameters*/
 	gcanbar = 0.0003;
 	_prop->param = _p;
 	_prop->param_size = 10;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[1]._pval = &prop_ion->param[2]; /* cao */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _can2_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 10, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 can /home/kanisorn/Downloads/147867/CA3Atrophy/can2.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 96520.0;
 static double R = 8.3134;
 static double KTOMV = .0853;
 static double *_t_alph;
 static double *_t_beth;
 static double *_t_alpm;
 static double *_t_betm;
static int _reset;
static char *modelname = "n-calcium channel";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 static double _n_betm(double);
 static double _n_alpm(double);
 static double _n_beth(double);
 static double _n_alph(double);
 
double h2 (  double _lcai ) {
   double _lh2;
 _lh2 = ki / ( ki + _lcai ) ;
   
return _lh2;
 }
 
static void _hoc_h2(void) {
  double _r;
   _r =  h2 (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double ghk (  double _lv , double _lci , double _lco ) {
   double _lghk;
 double _lnu , _lf ;
 _lf = KTF ( _threadargscomma_ celsius ) / 2.0 ;
   _lnu = _lv / _lf ;
   _lghk = - _lf * ( 1. - ( _lci / _lco ) * exp ( _lnu ) ) * efun ( _threadargscomma_ _lnu ) ;
   
return _lghk;
 }
 
static void _hoc_ghk(void) {
  double _r;
   _r =  ghk (  *getarg(1) , *getarg(2) , *getarg(3) );
 hoc_retpushx(_r);
}
 
double KTF (  double _lcelsius ) {
   double _lKTF;
 _lKTF = ( ( 25. / 293.15 ) * ( _lcelsius + 273.15 ) ) ;
   
return _lKTF;
 }
 
static void _hoc_KTF(void) {
  double _r;
   _r =  KTF (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double efun (  double _lz ) {
   double _lefun;
 if ( fabs ( _lz ) < 1e-4 ) {
     _lefun = 1.0 - _lz / 2.0 ;
     }
   else {
     _lefun = _lz / ( exp ( _lz ) - 1.0 ) ;
     }
   
return _lefun;
 }
 
static void _hoc_efun(void) {
  double _r;
   _r =  efun (  *getarg(1) );
 hoc_retpushx(_r);
}
 static double _mfac_alph, _tmin_alph;
 static void _check_alph();
 static void _check_alph() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_alph =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_alph)/200.; _mfac_alph = 1./_dx;
   for (_i=0, _x=_tmin_alph; _i < 201; _x += _dx, _i++) {
    _t_alph[_i] = _f_alph(_x);
   }
  }
 }

 double alph(double _lv){ _check_alph();
 return _n_alph(_lv);
 }

 static double _n_alph(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_alph(_lv); 
}
 _xi = _mfac_alph * (_lv - _tmin_alph);
 if (isnan(_xi)) {
  return _xi; }
 if (_xi <= 0.) {
 return _t_alph[0];
 }
 if (_xi >= 200.) {
 return _t_alph[200];
 }
 _i = (int) _xi;
 return _t_alph[_i] + (_xi - (double)_i)*(_t_alph[_i+1] - _t_alph[_i]);
 }

 
double _f_alph (  double _lv ) {
   double _lalph;
 _lalph = 1.6e-4 * exp ( - _lv / 48.4 ) ;
   
return _lalph;
 }
 
static void _hoc_alph(void) {
  double _r;
    _r =  alph (  *getarg(1) );
 hoc_retpushx(_r);
}
 static double _mfac_beth, _tmin_beth;
 static void _check_beth();
 static void _check_beth() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_beth =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_beth)/200.; _mfac_beth = 1./_dx;
   for (_i=0, _x=_tmin_beth; _i < 201; _x += _dx, _i++) {
    _t_beth[_i] = _f_beth(_x);
   }
  }
 }

 double beth(double _lv){ _check_beth();
 return _n_beth(_lv);
 }

 static double _n_beth(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_beth(_lv); 
}
 _xi = _mfac_beth * (_lv - _tmin_beth);
 if (isnan(_xi)) {
  return _xi; }
 if (_xi <= 0.) {
 return _t_beth[0];
 }
 if (_xi >= 200.) {
 return _t_beth[200];
 }
 _i = (int) _xi;
 return _t_beth[_i] + (_xi - (double)_i)*(_t_beth[_i+1] - _t_beth[_i]);
 }

 
double _f_beth (  double _lv ) {
   double _lbeth;
 _lbeth = 1.0 / ( exp ( ( - _lv + 39.0 ) / 10. ) + 1. ) ;
   
return _lbeth;
 }
 
static void _hoc_beth(void) {
  double _r;
    _r =  beth (  *getarg(1) );
 hoc_retpushx(_r);
}
 static double _mfac_alpm, _tmin_alpm;
 static void _check_alpm();
 static void _check_alpm() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_alpm =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_alpm)/200.; _mfac_alpm = 1./_dx;
   for (_i=0, _x=_tmin_alpm; _i < 201; _x += _dx, _i++) {
    _t_alpm[_i] = _f_alpm(_x);
   }
  }
 }

 double alpm(double _lv){ _check_alpm();
 return _n_alpm(_lv);
 }

 static double _n_alpm(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_alpm(_lv); 
}
 _xi = _mfac_alpm * (_lv - _tmin_alpm);
 if (isnan(_xi)) {
  return _xi; }
 if (_xi <= 0.) {
 return _t_alpm[0];
 }
 if (_xi >= 200.) {
 return _t_alpm[200];
 }
 _i = (int) _xi;
 return _t_alpm[_i] + (_xi - (double)_i)*(_t_alpm[_i+1] - _t_alpm[_i]);
 }

 
double _f_alpm (  double _lv ) {
   double _lalpm;
 _lalpm = 0.1967 * ( - 1.0 * _lv + 19.88 ) / ( exp ( ( - 1.0 * _lv + 19.88 ) / 10.0 ) - 1.0 ) ;
   
return _lalpm;
 }
 
static void _hoc_alpm(void) {
  double _r;
    _r =  alpm (  *getarg(1) );
 hoc_retpushx(_r);
}
 static double _mfac_betm, _tmin_betm;
 static void _check_betm();
 static void _check_betm() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_betm =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_betm)/200.; _mfac_betm = 1./_dx;
   for (_i=0, _x=_tmin_betm; _i < 201; _x += _dx, _i++) {
    _t_betm[_i] = _f_betm(_x);
   }
  }
 }

 double betm(double _lv){ _check_betm();
 return _n_betm(_lv);
 }

 static double _n_betm(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_betm(_lv); 
}
 _xi = _mfac_betm * (_lv - _tmin_betm);
 if (isnan(_xi)) {
  return _xi; }
 if (_xi <= 0.) {
 return _t_betm[0];
 }
 if (_xi >= 200.) {
 return _t_betm[200];
 }
 _i = (int) _xi;
 return _t_betm[_i] + (_xi - (double)_i)*(_t_betm[_i+1] - _t_betm[_i]);
 }

 
double _f_betm (  double _lv ) {
   double _lbetm;
 _lbetm = 0.046 * exp ( - _lv / 20.73 ) ;
   
return _lbetm;
 }
 
static void _hoc_betm(void) {
  double _r;
    _r =  betm (  *getarg(1) );
 hoc_retpushx(_r);
}
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / taum ;
   Dh = ( hinf - h ) / tauh ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taum )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tauh )) ;
  return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taum)))*(- ( ( ( minf ) ) / taum ) / ( ( ( ( - 1.0 ) ) ) / taum ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tauh)))*(- ( ( ( hinf ) ) / tauh ) / ( ( ( ( - 1.0 ) ) ) / tauh ) - h) ;
   }
  return 0;
}
 
static int  rates (  double _lv ) {
   double _la ;
 _la = alpm ( _threadargscomma_ _lv ) ;
   taum = 1.0 / ( _la + betm ( _threadargscomma_ _lv ) ) ;
   minf = _la * taum ;
   _la = alph ( _threadargscomma_ _lv ) ;
   tauh = 1.0 / ( _la + beth ( _threadargscomma_ _lv ) ) ;
   hinf = _la * tauh ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 2);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  h = h0;
  m = m0;
 {
   rates ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  cai = _ion_cai;
  cao = _ion_cao;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   gcan = gcanbar * m * m * h * h2 ( _threadargscomma_ cai ) ;
   ica = gcan * ghk ( _threadargscomma_ v , cai , cao ) ;
   }
 _current += ica;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  cai = _ion_cai;
  cao = _ion_cao;
 _g = _nrn_current(_v + .001);
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_v);
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  cai = _ion_cai;
  cao = _ion_cao;
 { error =  states();
 if(error){fprintf(stderr,"at line 51 in file can2.mod:\n	SOLVE states METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
   _t_alph = makevector(201*sizeof(double));
   _t_beth = makevector(201*sizeof(double));
   _t_alpm = makevector(201*sizeof(double));
   _t_betm = makevector(201*sizeof(double));
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/home/kanisorn/Downloads/147867/CA3Atrophy/can2.mod";
static const char* nmodl_file_text = 
  "TITLE n-calcium channel\n"
  ": n-type calcium channel\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "\n"
  "	FARADAY = 96520 (coul)\n"
  "	R = 8.3134 (joule/degC)\n"
  "	KTOMV = .0853 (mV/degC)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	v (mV)\n"
  "	celsius 		(degC)\n"
  "	gcanbar=.0003 (mho/cm2)\n"
  "	ki=.001 (mM)\n"
  "	cai=5.e-5 (mM)\n"
  "	cao = 10  (mM)\n"
  "}\n"
  "\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX can\n"
  "	USEION ca READ cai,cao WRITE ica\n"
  "        RANGE gcanbar       \n"
  "        GLOBAL hinf,minf,taum,tauh\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	m h \n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	ica (mA/cm2)\n"
  "        gcan  (mho/cm2) \n"
  "        minf\n"
  "        hinf\n"
  "        taum\n"
  "        tauh\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "        rates(v)\n"
  "        m = minf\n"
  "        h = hinf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	gcan = gcanbar*m*m*h*h2(cai)\n"
  "	ica = gcan*ghk(v,cai,cao)\n"
  "\n"
  "}\n"
  "\n"
  "UNITSOFF\n"
  "FUNCTION h2(cai(mM)) {\n"
  "	h2 = ki/(ki+cai)\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {\n"
  "        LOCAL nu,f\n"
  "\n"
  "        f = KTF(celsius)/2\n"
  "        nu = v/f\n"
  "        ghk=-f*(1. - (ci/co)*exp(nu))*efun(nu)\n"
  "}\n"
  "\n"
  "FUNCTION KTF(celsius (degC)) (mV) {\n"
  "        KTF = ((25./293.15)*(celsius + 273.15))\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION efun(z) {\n"
  "	if (fabs(z) < 1e-4) {\n"
  "		efun = 1 - z/2\n"
  "	}else{\n"
  "		efun = z/(exp(z) - 1)\n"
  "	}\n"
  "}\n"
  "\n"
  "FUNCTION alph(v(mV)) {\n"
  "	TABLE FROM -150 TO 150 WITH 200\n"
  "	alph = 1.6e-4*exp(-v/48.4)\n"
  "}\n"
  "\n"
  "FUNCTION beth(v(mV)) {\n"
  "        TABLE FROM -150 TO 150 WITH 200\n"
  "	beth = 1/(exp((-v+39.0)/10.)+1.)\n"
  "}\n"
  "\n"
  "FUNCTION alpm(v(mV)) {\n"
  "	TABLE FROM -150 TO 150 WITH 200\n"
  "	alpm = 0.1967*(-1.0*v+19.88)/(exp((-1.0*v+19.88)/10.0)-1.0)\n"
  "}\n"
  "\n"
  "FUNCTION betm(v(mV)) {\n"
  "	TABLE FROM -150 TO 150 WITH 200\n"
  "	betm = 0.046*exp(-v/20.73)\n"
  "}\n"
  "\n"
  "UNITSON\n"
  "\n"
  "DERIVATIVE states {     : exact when v held constant; integrates over dt step\n"
  "        rates(v)\n"
  "        m' = (minf - m)/taum\n"
  "        h' = (hinf - h)/tauh\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v (mV)) { :callable from hoc\n"
  "        LOCAL a\n"
  "        a = alpm(v)\n"
  "        taum = 1/(a + betm(v))\n"
  "        minf = a*taum\n"
  "        a = alph(v)\n"
  "        tauh = 1/(a + beth(v))\n"
  "        hinf = a*tauh\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
