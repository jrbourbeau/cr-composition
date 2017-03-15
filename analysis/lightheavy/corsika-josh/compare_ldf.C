{

  TCanvas *c = new TCanvas("c","C",900,800);
  c->SetLogy();
  gStyle->SetOptStat(0);

  TFile *tf1 = new TFile("proton_1PeV_mu_0.2GeV.root","READ");
  TProfile *tp = (TProfile*)tf1->Get("hemp;1");
  tp->SetName("proton");
  tp->SetTitle("Lateral Distribution of Energy in IceTop;Radius [m];Energy/Area [GeV/m^{2}]");
  tp->SetLineColor(kRed);
//  tp->GetYaxis()->SetRangeUser(0,250);
  tp->GetXaxis()->SetTitleOffset(1.3);
  tp->GetYaxis()->SetTitleOffset(1.3);

  TFile *tf2 = new TFile("iron_1PeV_mu_0.2GeV.root","READ");
  TProfile *tfe = (TProfile*)tf2->Get("hemp;1");
  tfe->SetLineColor(kBlue);

  tp->Draw();
  tfe->Draw("same");

  double sump = 0;
  double sumfe = 0;

  int N = tp->GetNbinsX();
  for (int i=1; i<=N; i++) {
    sump  += tp->GetBinContent(i);
    sumfe += tfe->GetBinContent(i);
  }
  printf(" muons in proton shower: %.2f\n", sump);
  printf(" muons in iron shower: %.2f (%.2f)\n", sumfe, (sumfe/sump));

  TLegend *tl = new TLegend(0.55,0.75,0.9,0.9);
  tl->AddEntry(tp,"Vertical 1 PeV Proton","l");
  tl->AddEntry(tfe,"Vertical 1 PeV Iron","l");
  tl->Draw();

  double r = 250;
  double ri = tp->FindBin(r);
  printf(" R = %.2f\n", tp->GetBinCenter(ri));
  printf("   proton %.2f\n", tp->GetBinContent(ri));
  printf("     iron %.2f (%.2f)\n", tfe->GetBinContent(ri), tfe->GetBinContent(ri)/tp->GetBinContent(ri));

  c->SaveAs("ldf.png");
}
