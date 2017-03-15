{

  TCanvas *c = new TCanvas("c","C",900,800);
  gStyle->SetOptStat(0);

  TFile *tf1 = new TFile("proton_1PeV_mu_0.2GeV.root","READ");
  TProfile *tp = (TProfile*)tf1->Get("hnmup;1");
  tp->SetName("proton");
  tp->SetTitle("Lateral Distribution of Muons in IceTop;Radius [m];Number of Muons (E > 0.2 GeV)");
  tp->SetLineColor(kRed);
  tp->GetYaxis()->SetRangeUser(0,265);
  tp->GetXaxis()->SetTitleOffset(1.3);
  tp->GetYaxis()->SetTitleOffset(1.3);

  TFile *tf2 = new TFile("iron_1PeV_mu_0.2GeV.root","READ");
  TProfile *tfe = (TProfile*)tf2->Get("hnmup;1");
  tfe->SetLineColor(kBlue);

  tp->Draw();
  tfe->Draw("same");

  double sump = 0;
  double sumfe = 0;

  // icetop params
  double Rtank = 0.91;
  double Ntank = 162;
  double Atank = 3.14159*Rtank*Rtank;
  double Atanks = Ntank * Atank;
  double fA = Atanks/1000./1000.; // fractional area coverage

  int N = tp->GetNbinsX();
  double Rmin = 178;
  double Rmax = tp->GetBinCenter(N)+tp->GetBinWidth(N)/2.;
  double Atot = 3.14159*(Rmax*Rmax - Rmin*Rmin);
  for (int i=1; i<=N; i++) {
    if (tp->GetBinCenter(i)-tp->GetBinWidth(i)/2. > Rmin) {
      sump  += tp->GetBinContent(i);
      sumfe += tfe->GetBinContent(i);
    }
  }
  printf(" Between radii %.2f - %.2f:\n", Rmin, Rmax);
  printf("   total area %.2f m^2, covered area %.2f m^2 (%.2e)\n",Atot,Atot*fA,fA);
  printf("   muons in proton shower:    total %.2f\n", sump);
  printf("                           detected %.2f\n", sump*fA);
  printf("   muons in iron shower:      total %.2f\n", sumfe);
  printf("                           detected %.2f\n", sumfe*fA);
  printf("   ratio iron/proton: %.2f\n", sumfe/sump);

  gPad->Update();
  TLine *line = new TLine();
  line->SetLineStyle(7);
  line->DrawLine(178,gPad->GetUymin(),178,gPad->GetUymax());

  double ay = gPad->GetUymin() + 0.905*(gPad->GetUymax()-gPad->GetUymin());
  TArrow *arrow = new TArrow(178, ay, 198, ay, 0.01, "|>");
  arrow->Draw();

  TLegend *tl = new TLegend(0.55,0.75,0.9,0.9);
  tl->AddEntry(tp,"Vertical 1 PeV Proton","l");
  tl->AddEntry(tfe,"Vertical 1 PeV Iron","l");
  tl->Draw();

  double r = 250;
  double ri = tp->FindBin(r);
  printf(" R = %.2f\n", tp->GetBinCenter(ri));
  printf("   proton %.2f\n", tp->GetBinContent(ri));
  printf("     iron %.2f (%.2f)\n", tfe->GetBinContent(ri), tfe->GetBinContent(ri)/tp->GetBinContent(ri));

  c->SaveAs("muons.png");
}
