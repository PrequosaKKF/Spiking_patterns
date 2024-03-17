#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _ampa_reg(void);
extern void _borgkm_reg(void);
extern void _cadiv_reg(void);
extern void _cagk_reg(void);
extern void _cal2_reg(void);
extern void _can2_reg(void);
extern void _cat_reg(void);
extern void _h_reg(void);
extern void _kad_reg(void);
extern void _kahp_reg(void);
extern void _kap_reg(void);
extern void _kdr_reg(void);
extern void _nahh_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"ampa.mod\"");
    fprintf(stderr, " \"borgkm.mod\"");
    fprintf(stderr, " \"cadiv.mod\"");
    fprintf(stderr, " \"cagk.mod\"");
    fprintf(stderr, " \"cal2.mod\"");
    fprintf(stderr, " \"can2.mod\"");
    fprintf(stderr, " \"cat.mod\"");
    fprintf(stderr, " \"h.mod\"");
    fprintf(stderr, " \"kad.mod\"");
    fprintf(stderr, " \"kahp.mod\"");
    fprintf(stderr, " \"kap.mod\"");
    fprintf(stderr, " \"kdr.mod\"");
    fprintf(stderr, " \"nahh.mod\"");
    fprintf(stderr, "\n");
  }
  _ampa_reg();
  _borgkm_reg();
  _cadiv_reg();
  _cagk_reg();
  _cal2_reg();
  _can2_reg();
  _cat_reg();
  _h_reg();
  _kad_reg();
  _kahp_reg();
  _kap_reg();
  _kdr_reg();
  _nahh_reg();
}

#if defined(__cplusplus)
}
#endif
