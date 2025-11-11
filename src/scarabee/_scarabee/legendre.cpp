#include <moc/quadrature/legendre.hpp>

#include <cmath>

namespace scarabee {

// N = 2
template <>
const std::array<double, 1> Legendre<2>::sin_ = {
    std::sqrt(1. - std::pow(5.77350269189625764509148780501957456e-01, 2.))};
template <>
const std::array<double, 1> Legendre<2>::invs_sin_ = {1. / sin_[0]};
template <>
const std::array<double, 1> Legendre<2>::wgt_ = {
    1.00000000000000000000000000000000000e+00};
template <>
const std::array<double, 1> Legendre<2>::wsin_ = {wgt_[0] * sin_[0]};
template <>
const std::array<double, 1> Legendre<2>::polar_angle_ = {std::asin(sin_[0])};

// N = 4
template <>
const std::array<double, 2> Legendre<4>::sin_ = {
    std::sqrt(1. - std::pow(3.39981043584856264802665759103244687e-01, 2.)),
    std::sqrt(1. - std::pow(8.61136311594052575223946488892809505e-01, 2.))};
template <>
const std::array<double, 2> Legendre<4>::invs_sin_ = {1. / sin_[0],
                                                      1. / sin_[1]};
template <>
const std::array<double, 2> Legendre<4>::wgt_ = {
    6.52145154862546142626936050778000593e-01,
    3.47854845137453857373063949221999407e-01};
template <>
const std::array<double, 2> Legendre<4>::wsin_ = {wgt_[0] * sin_[0],
                                                  wgt_[1] * sin_[1]};
template <>
const std::array<double, 2> Legendre<4>::polar_angle_ = {std::asin(sin_[0]),
                                                         std::asin(sin_[1])};

// N = 6
template <>
const std::array<double, 3> Legendre<6>::sin_ = {
    std::sqrt(1. - std::pow(2.38619186083196908630501721680711935e-01, 2.)),
    std::sqrt(1. - std::pow(6.61209386466264513661399595019905347e-01, 2.)),
    std::sqrt(1. - std::pow(9.32469514203152027812301554493994609e-01, 2.))};
template <>
const std::array<double, 3> Legendre<6>::invs_sin_ = {
    1. / sin_[0], 1. / sin_[1], 1. / sin_[2]};
template <>
const std::array<double, 3> Legendre<6>::wgt_ = {
    4.67913934572691047389870343989550995e-01,
    3.60761573048138607569833513837716112e-01,
    1.71324492379170345040296142172732894e-01};
template <>
const std::array<double, 3> Legendre<6>::wsin_ = {
    wgt_[0] * sin_[0], wgt_[1] * sin_[1], wgt_[2] * sin_[2]};
template <>
const std::array<double, 3> Legendre<6>::polar_angle_ = {
    std::asin(sin_[0]), std::asin(sin_[1]), std::asin(sin_[2])};

// N = 8
template <>
const std::array<double, 4> Legendre<8>::sin_ = {
    std::sqrt(1. - std::pow(1.83434642495649804939476142360183981e-01, 2.)),
    std::sqrt(1. - std::pow(5.25532409916328985817739049189246349e-01, 2.)),
    std::sqrt(1. - std::pow(7.96666477413626739591553936475830437e-01, 2.)),
    std::sqrt(1. - std::pow(9.60289856497536231683560868569472990e-01, 2.))};
template <>
const std::array<double, 4> Legendre<8>::invs_sin_ = {
    1. / sin_[0], 1. / sin_[1], 1. / sin_[2], 1. / sin_[3]};
template <>
const std::array<double, 4> Legendre<8>::wgt_ = {
    3.62683783378361982965150449277195612e-01,
    3.13706645877887287337962201986601313e-01,
    2.22381034453374470544355994426240884e-01,
    1.01228536290376259152531354309962190e-01};
template <>
const std::array<double, 4> Legendre<8>::wsin_ = {
    wgt_[0] * sin_[0], wgt_[1] * sin_[1], wgt_[2] * sin_[2], wgt_[3] * sin_[3]};
template <>
const std::array<double, 4> Legendre<8>::polar_angle_ = {
    std::asin(sin_[0]), std::asin(sin_[1]), std::asin(sin_[2]),
    std::asin(sin_[3])};

// N = 10
template <>
const std::array<double, 5> Legendre<10>::sin_ = {
    std::sqrt(1. - std::pow(1.48874338981631210884826001129719985e-01, 2.)),
    std::sqrt(1. - std::pow(4.33395394129247190799265943165784162e-01, 2.)),
    std::sqrt(1. - std::pow(6.79409568299024406234327365114873576e-01, 2.)),
    std::sqrt(1. - std::pow(8.65063366688984510732096688423493049e-01, 2.)),
    std::sqrt(1. - std::pow(9.73906528517171720077964012084452053e-01, 2.))};
template <>
const std::array<double, 5> Legendre<10>::invs_sin_ = {
    1. / sin_[0], 1. / sin_[1], 1. / sin_[2], 1. / sin_[3], 1. / sin_[4]};
template <>
const std::array<double, 5> Legendre<10>::wgt_ = {
    2.95524224714752870173892994651338329e-01,
    2.69266719309996355091226921569469353e-01,
    2.19086362515982043995534934228163192e-01,
    1.49451349150580593145776339657697332e-01,
    6.66713443086881375935688098933317929e-02};
template <>
const std::array<double, 5> Legendre<10>::wsin_ = {
    wgt_[0] * sin_[0], wgt_[1] * sin_[1], wgt_[2] * sin_[2], wgt_[3] * sin_[3],
    wgt_[4] * sin_[4]};
template <>
const std::array<double, 5> Legendre<10>::polar_angle_ = {
    std::asin(sin_[0]), std::asin(sin_[1]), std::asin(sin_[2]),
    std::asin(sin_[3]), std::asin(sin_[4])};

// N = 12
template <>
const std::array<double, 6> Legendre<12>::sin_ = {
    std::sqrt(1. - std::pow(1.25233408511468915472441369463853130e-01, 2.)),
    std::sqrt(1. - std::pow(3.67831498998180193752691536643717561e-01, 2.)),
    std::sqrt(1. - std::pow(5.87317954286617447296702418940534280e-01, 2.)),
    std::sqrt(1. - std::pow(7.69902674194304687036893833212818076e-01, 2.)),
    std::sqrt(1. - std::pow(9.04117256370474856678465866119096193e-01, 2.)),
    std::sqrt(1. - std::pow(9.81560634246719250690549090149280823e-01, 2.))};
template <>
const std::array<double, 6> Legendre<12>::invs_sin_ = {
    1. / sin_[0], 1. / sin_[1], 1. / sin_[2],
    1. / sin_[3], 1. / sin_[4], 1. / sin_[5]};
template <>
const std::array<double, 6> Legendre<12>::wgt_ = {
    2.49147045813402785000562436042951211e-01,
    2.33492536538354808760849898924878056e-01,
    2.03167426723065921749064455809798377e-01,
    1.60078328543346226334652529543359072e-01,
    1.06939325995318430960254718193996224e-01,
    4.71753363865118271946159614850170603e-02};
template <>
const std::array<double, 6> Legendre<12>::wsin_ = {
    wgt_[0] * sin_[0], wgt_[1] * sin_[1], wgt_[2] * sin_[2],
    wgt_[3] * sin_[3], wgt_[4] * sin_[4], wgt_[5] * sin_[5]};
template <>
const std::array<double, 6> Legendre<12>::polar_angle_ = {
    std::asin(sin_[0]), std::asin(sin_[1]), std::asin(sin_[2]),
    std::asin(sin_[3]), std::asin(sin_[4]), std::asin(sin_[5])};

// N = 16
template <>
const std::array<double, 8> Legendre<16>::sin_ = {
    std::sqrt(1. - std::pow(9.50125098376374401853e-02, 2.)),
    std::sqrt(1. - std::pow(2.81603550779258913230e-01, 2.)),
    std::sqrt(1. - std::pow(4.58016777657227386342e-01, 2.)),
    std::sqrt(1. - std::pow(6.17876244402643748447e-01, 2.)),
    std::sqrt(1. - std::pow(7.55404408355003033895e-01, 2.)),
    std::sqrt(1. - std::pow(8.65631202387831743880e-01, 2.)),
    std::sqrt(1. - std::pow(9.44575023073232576078e-01, 2.)),
    std::sqrt(1. - std::pow(9.89400934991649932596e-01, 2.))};
template <>
const std::array<double, 8> Legendre<16>::invs_sin_ = {
    1. / sin_[0], 1. / sin_[1], 1. / sin_[2], 1. / sin_[3],
    1. / sin_[4], 1. / sin_[5], 1. / sin_[6], 1. / sin_[7]};
template <>
const std::array<double, 8> Legendre<16>::wgt_ = {
    1.89450610455068496285e-01, 1.82603415044923588867e-01,
    1.69156519395002538189e-01, 1.49595988816576732082e-01,
    1.24628971255533872052e-01, 9.51585116824927848099e-02,
    6.22535239386478928628e-02, 2.71524594117540948518e-02};
template <>
const std::array<double, 8> Legendre<16>::wsin_ = {
    wgt_[0] * sin_[0], wgt_[1] * sin_[1], wgt_[2] * sin_[2], wgt_[3] * sin_[3],
    wgt_[4] * sin_[4], wgt_[5] * sin_[5], wgt_[6] * sin_[6], wgt_[7] * sin_[7]};
template <>
const std::array<double, 8> Legendre<16>::polar_angle_ = {
    std::asin(sin_[0]), std::asin(sin_[1]), std::asin(sin_[2]),
    std::asin(sin_[3]), std::asin(sin_[4]), std::asin(sin_[5]),
    std::asin(sin_[6]), std::asin(sin_[7])};

// N = 32
template <>
const std::array<double, 16> Legendre<32>::sin_ = {
    std::sqrt(1. - std::pow(4.83076656877383162348e-02, 2.)),
    std::sqrt(1. - std::pow(1.44471961582796493485e-01, 2.)),
    std::sqrt(1. - std::pow(2.39287362252137074545e-01, 2.)),
    std::sqrt(1. - std::pow(3.31868602282127649780e-01, 2.)),
    std::sqrt(1. - std::pow(4.21351276130635345364e-01, 2.)),
    std::sqrt(1. - std::pow(5.06899908932229390024e-01, 2.)),
    std::sqrt(1. - std::pow(5.87715757240762329041e-01, 2.)),
    std::sqrt(1. - std::pow(6.63044266930215200975e-01, 2.)),
    std::sqrt(1. - std::pow(7.32182118740289680387e-01, 2.)),
    std::sqrt(1. - std::pow(7.94483795967942406963e-01, 2.)),
    std::sqrt(1. - std::pow(8.49367613732569970134e-01, 2.)),
    std::sqrt(1. - std::pow(8.96321155766052123965e-01, 2.)),
    std::sqrt(1. - std::pow(9.34906075937739689171e-01, 2.)),
    std::sqrt(1. - std::pow(9.64762255587506430774e-01, 2.)),
    std::sqrt(1. - std::pow(9.85611511545268335400e-01, 2.)),
    std::sqrt(1. - std::pow(9.97263861849481563545e-01, 2.))};
template <>
const std::array<double, 16> Legendre<32>::invs_sin_ = {
    1. / sin_[0],  1. / sin_[1],  1. / sin_[2],  1. / sin_[3],
    1. / sin_[4],  1. / sin_[5],  1. / sin_[6],  1. / sin_[7],
    1. / sin_[8],  1. / sin_[9],  1. / sin_[10], 1. / sin_[11],
    1. / sin_[12], 1. / sin_[13], 1. / sin_[14], 1. / sin_[15]};
template <>
const std::array<double, 16> Legendre<32>::wgt_ = {
    9.65400885147278005668e-02, 9.56387200792748594191e-02,
    9.38443990808045656392e-02, 9.11738786957638847129e-02,
    8.76520930044038111428e-02, 8.33119242269467552222e-02,
    7.81938957870703064717e-02, 7.23457941088485062254e-02,
    6.58222227763618468377e-02, 5.86840934785355471453e-02,
    5.09980592623761761962e-02, 4.28358980222266806569e-02,
    3.42738629130214331027e-02, 2.53920653092620594558e-02,
    1.62743947309056706052e-02, 7.01861000947009660041e-03};
template <>
const std::array<double, 16> Legendre<32>::wsin_ = {
    wgt_[0] * sin_[0],   wgt_[1] * sin_[1],   wgt_[2] * sin_[2],
    wgt_[3] * sin_[3],   wgt_[4] * sin_[4],   wgt_[5] * sin_[5],
    wgt_[6] * sin_[6],   wgt_[7] * sin_[7],   wgt_[8] * sin_[8],
    wgt_[9] * sin_[9],   wgt_[10] * sin_[10], wgt_[11] * sin_[11],
    wgt_[12] * sin_[12], wgt_[13] * sin_[13], wgt_[14] * sin_[14],
    wgt_[15] * sin_[15]};
template <>
const std::array<double, 16> Legendre<32>::polar_angle_ = {
    std::asin(sin_[0]),  std::asin(sin_[1]),  std::asin(sin_[2]),
    std::asin(sin_[3]),  std::asin(sin_[4]),  std::asin(sin_[5]),
    std::asin(sin_[6]),  std::asin(sin_[7]),  std::asin(sin_[8]),
    std::asin(sin_[9]),  std::asin(sin_[10]), std::asin(sin_[11]),
    std::asin(sin_[12]), std::asin(sin_[13]), std::asin(sin_[14]),
    std::asin(sin_[15])};

// N = 64
template <>
const std::array<double, 32> Legendre<64>::sin_ = {
    std::sqrt(1. - std::pow(2.43502926634244325090e-02, 2.)),
    std::sqrt(1. - std::pow(7.29931217877990394495e-02, 2.)),
    std::sqrt(1. - std::pow(1.21462819296120554470e-01, 2.)),
    std::sqrt(1. - std::pow(1.69644420423992818037e-01, 2.)),
    std::sqrt(1. - std::pow(2.17423643740007084150e-01, 2.)),
    std::sqrt(1. - std::pow(2.64687162208767416374e-01, 2.)),
    std::sqrt(1. - std::pow(3.11322871990210956158e-01, 2.)),
    std::sqrt(1. - std::pow(3.57220158337668115950e-01, 2.)),
    std::sqrt(1. - std::pow(4.02270157963991603696e-01, 2.)),
    std::sqrt(1. - std::pow(4.46366017253464087985e-01, 2.)),
    std::sqrt(1. - std::pow(4.89403145707052957479e-01, 2.)),
    std::sqrt(1. - std::pow(5.31279464019894545658e-01, 2.)),
    std::sqrt(1. - std::pow(5.71895646202634034284e-01, 2.)),
    std::sqrt(1. - std::pow(6.11155355172393250249e-01, 2.)),
    std::sqrt(1. - std::pow(6.48965471254657339858e-01, 2.)),
    std::sqrt(1. - std::pow(6.85236313054233242564e-01, 2.)),
    std::sqrt(1. - std::pow(7.19881850171610826849e-01, 2.)),
    std::sqrt(1. - std::pow(7.52819907260531896612e-01, 2.)),
    std::sqrt(1. - std::pow(7.83972358943341407610e-01, 2.)),
    std::sqrt(1. - std::pow(8.13265315122797559742e-01, 2.)),
    std::sqrt(1. - std::pow(8.40629296252580362752e-01, 2.)),
    std::sqrt(1. - std::pow(8.65999398154092819761e-01, 2.)),
    std::sqrt(1. - std::pow(8.89315445995114105853e-01, 2.)),
    std::sqrt(1. - std::pow(9.10522137078502805756e-01, 2.)),
    std::sqrt(1. - std::pow(9.29569172131939575821e-01, 2.)),
    std::sqrt(1. - std::pow(9.46411374858402816062e-01, 2.)),
    std::sqrt(1. - std::pow(9.61008799652053718919e-01, 2.)),
    std::sqrt(1. - std::pow(9.73326827789910963742e-01, 2.)),
    std::sqrt(1. - std::pow(9.83336253884625956931e-01, 2.)),
    std::sqrt(1. - std::pow(9.91013371476744320739e-01, 2.)),
    std::sqrt(1. - std::pow(9.96340116771955279347e-01, 2.)),
    std::sqrt(1. - std::pow(9.99305041735772139457e-01, 2.))};
template <>
const std::array<double, 32> Legendre<64>::invs_sin_ = {
    1. / sin_[0],  1. / sin_[1],  1. / sin_[2],  1. / sin_[3],  1. / sin_[4],
    1. / sin_[5],  1. / sin_[6],  1. / sin_[7],  1. / sin_[8],  1. / sin_[9],
    1. / sin_[10], 1. / sin_[11], 1. / sin_[12], 1. / sin_[13], 1. / sin_[14],
    1. / sin_[15], 1. / sin_[16], 1. / sin_[17], 1. / sin_[18], 1. / sin_[19],
    1. / sin_[20], 1. / sin_[21], 1. / sin_[22], 1. / sin_[23], 1. / sin_[24],
    1. / sin_[25], 1. / sin_[26], 1. / sin_[27], 1. / sin_[28], 1. / sin_[29],
    1. / sin_[30], 1. / sin_[31]};
template <>
const std::array<double, 32> Legendre<64>::wgt_ = {
    4.86909570091397203834e-02, 4.85754674415034269348e-02,
    4.83447622348029571698e-02, 4.79993885964583077281e-02,
    4.75401657148303086623e-02, 4.69681828162100173253e-02,
    4.62847965813144172960e-02, 4.54916279274181444798e-02,
    4.45905581637565630601e-02, 4.35837245293234533768e-02,
    4.24735151236535890073e-02, 4.12625632426235286102e-02,
    3.99537411327203413867e-02, 3.85501531786156291290e-02,
    3.70551285402400460404e-02, 3.54722132568823838107e-02,
    3.38051618371416093916e-02, 3.20579283548515535855e-02,
    3.02346570724024788680e-02, 2.83396726142594832275e-02,
    2.63774697150546586717e-02, 2.43527025687108733382e-02,
    2.22701738083832541593e-02, 2.01348231535302093723e-02,
    1.79517157756973430850e-02, 1.57260304760247193220e-02,
    1.34630478967186425981e-02, 1.11681394601311288186e-02,
    8.84675982636394772303e-03, 6.50445796897836285612e-03,
    4.14703326056246763529e-03, 1.78328072169643294730e-03};
template <>
const std::array<double, 32> Legendre<64>::wsin_ = {
    wgt_[0] * sin_[0],   wgt_[1] * sin_[1],   wgt_[2] * sin_[2],
    wgt_[3] * sin_[3],   wgt_[4] * sin_[4],   wgt_[5] * sin_[5],
    wgt_[6] * sin_[6],   wgt_[7] * sin_[7],   wgt_[8] * sin_[8],
    wgt_[9] * sin_[9],   wgt_[10] * sin_[10], wgt_[11] * sin_[11],
    wgt_[12] * sin_[12], wgt_[13] * sin_[13], wgt_[14] * sin_[14],
    wgt_[15] * sin_[15], wgt_[16] * sin_[16], wgt_[17] * sin_[17],
    wgt_[18] * sin_[18], wgt_[19] * sin_[19], wgt_[20] * sin_[20],
    wgt_[21] * sin_[21], wgt_[22] * sin_[22], wgt_[23] * sin_[23],
    wgt_[24] * sin_[24], wgt_[25] * sin_[25], wgt_[26] * sin_[26],
    wgt_[27] * sin_[27], wgt_[28] * sin_[28], wgt_[29] * sin_[29],
    wgt_[30] * sin_[30], wgt_[31] * sin_[31]};
template <>
const std::array<double, 32> Legendre<64>::polar_angle_ = {
    std::asin(sin_[0]),  std::asin(sin_[1]),  std::asin(sin_[2]),
    std::asin(sin_[3]),  std::asin(sin_[4]),  std::asin(sin_[5]),
    std::asin(sin_[6]),  std::asin(sin_[7]),  std::asin(sin_[8]),
    std::asin(sin_[9]),  std::asin(sin_[10]), std::asin(sin_[11]),
    std::asin(sin_[12]), std::asin(sin_[13]), std::asin(sin_[14]),
    std::asin(sin_[15]), std::asin(sin_[16]), std::asin(sin_[17]),
    std::asin(sin_[18]), std::asin(sin_[19]), std::asin(sin_[20]),
    std::asin(sin_[21]), std::asin(sin_[22]), std::asin(sin_[23]),
    std::asin(sin_[24]), std::asin(sin_[25]), std::asin(sin_[26]),
    std::asin(sin_[27]), std::asin(sin_[28]), std::asin(sin_[29]),
    std::asin(sin_[30]), std::asin(sin_[31])};

}  // namespace scarabee
