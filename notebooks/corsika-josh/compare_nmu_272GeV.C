{

  TCanvas *c = new TCanvas("c","C",900,800);
  gStyle->SetOptStat(0);

  TFile *tf1 = new TFile("proton_1PeV_mu_272GeV.root","READ");
  TProfile *tp = (TProfile*)tf1->Get("hnmup;1");
  tp->SetName("proton");
  tp->SetTitle("Lateral Distribution of Muons in IceTop;Radius [m];Number of Muons (E > 272 GeV)");
  tp->SetLineColor(kRed);
  tp->GetYaxis()->SetRangeUser(0,30);
  tp->GetXaxis()->SetRangeUser(0,100);
  tp->GetXaxis()->SetTitleOffset(1.3);
  tp->GetYaxis()->SetTitleOffset(1.3);

  TFile *tf2 = new TFile("iron_1PeV_mu_272GeV.root","READ");
  TProfile *tfe = (TProfile*)tf2->Get("hnmup;1");
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
  printf("   muons in proton shower:    total %.2f\n", sump);
  printf("   muons in iron shower:      total %.2f\n", sumfe);
  printf("   ratio iron/proton: %.2f\n", sumfe/sump);

  TLegend *tl = new TLegend(0.55,0.75,0.9,0.9);
  tl->AddEntry(tp,"Vertical 1 PeV Proton","l");
  tl->AddEntry(tfe,"Vertical 1 PeV Iron","l");
  tl->Draw();

  c->SaveAs("muons_272GeV.png");
}
